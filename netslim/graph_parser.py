import re
import torch

CHANNEL_DIM = 1
NORM_LAYER_KEYWORDS = ["batch_norm", "group_norm", "instance_norm"]
PRUNABLE_LAYER_KEYWORDS = ["convolution", "addmm"]  # does not support groups > 1 for conv
PASS_KEYWORDS = ["relu", "leaky_relu", "sigmoid", "tanh",
                 "pool", "pad", "dropout",
                 "view", "flatten"]  # and more .. does not support concat
OTHER_OP_KEYWORDS = ["cat"]
OTHER_PRIM_KEYWORDS = ["ListConstruct"]  # for cat
NO_EFFECT_KEYWORDS = ["size", ]

scope_pattern = re.compile(r".+, scope: (.+)")
module_pattern = re.compile(r"\[(\w+)\]")
output_pattern = re.compile(r"(%.+) : .*, scope: .+")
input_pattern = re.compile(r".+ = aten::\w+\((%.+),*\), scope: .+")
prim_input_pattern = re.compile(r".+ = prim::\w+\((%.+),*\), scope: .+")
shape_pattern = re.compile(r"%.+ : \w+\((.+)\) = aten::\w+\(%.+\), scope: .+")
int_pattern = re.compile(r"[1-9]+")
view_pattern = re.compile(r"aten::.*view.*")

norm_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in NORM_LAYER_KEYWORDS])
    )
)
prunable_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in PRUNABLE_LAYER_KEYWORDS])
    )
)
pass_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in PASS_KEYWORDS])
    )
)
allowed_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in
                  PRUNABLE_LAYER_KEYWORDS +
                  PASS_KEYWORDS +
                  NO_EFFECT_KEYWORDS])
    )
)
common_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in
                  NORM_LAYER_KEYWORDS +
                  PRUNABLE_LAYER_KEYWORDS +
                  PASS_KEYWORDS +
                  OTHER_OP_KEYWORDS]
                 +["prim::\w*{}\w*".format(_) for _ in
                   OTHER_PRIM_KEYWORDS])
    )
)
tensor_op_pattern = re.compile(r".*= (aten::\w+)\(.*, scope: .+")
add_op_pattern = re.compile(r".*= (aten::add_)\(.*, scope: .+")


def get_node_str(node):
    return repr(node).split(" # ")[0]


def parse_module_name(x):
    scope_found = scope_pattern.findall(x)
    module_name = ''
    if scope_found:
        tokens = scope_found[0].split('/')[1:]
        module_name = '.'.join([module_pattern.findall(_)[0] for _ in tokens])
    return module_name


def parse_output_name(x):
    return output_pattern.findall(x)[0]


def parse_input_names(x):
    result = input_pattern.findall(x)
    if not result:
        result = prim_input_pattern.findall(x)
    return result[0].split(", ")


def parse_output_shape(x):
    sizes = shape_pattern.findall(x)[0].split(", ")
    for s in sizes:
        if not int_pattern.match(s):
            return None
    return [int(_) for _ in sizes]


# assume for a normalization layer, it has only one input/output
def get_norm_layer_io(graph):
    out2nl = {}
    in2nl = {}
    bn_names = []
    for node in graph.nodes():
        node_str = get_node_str(node)
        if norm_layer_pattern.match(node_str):
            bn_name = parse_module_name(node_str)
            output = parse_output_name(node_str)
            input = parse_input_names(node_str)[0]
            out2nl[output] = bn_name
            in2nl[input] = bn_name
            bn_names.append(bn_name)
    return out2nl, in2nl, bn_names


def reverse_search_dict(val, target_dict):
    return [k for k, v in target_dict.items() if v == val]


# check for tensor operation layer and prim::ListConstruct, which is used by cat operation
def get_input_count(graph):
    input_count = {}
    for node in graph.nodes():
        node_str = get_node_str(node)
        matches = common_layer_pattern.findall(node_str)
        if matches:
            input_names = parse_input_names(node_str)
            for input_name in input_names:
                if input_name not in input_count:
                    input_count[input_name] = 1
                else:
                    input_count[input_name] += 1
    return input_count


def get_pruning_layers(model, input_shape, device=None):
    """parse the model graph, and generate mapping to BNs

    Arguments:
        model (pytorch model): the model instance
        input_shape (tuple): shape of the input tensor

    Returns:
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
    """

    # 0 trace graph with torch scripts
    inputs = torch.randn(2, *input_shape)   # 2 for BatchNorm1d
    if device:
        inputs = inputs.to(device)
    trace, _ = torch.jit.get_trace_graph(model, args=(inputs,))
    graph = trace.graph()
    input_count = get_input_count(graph)

    # 1 get norm layers and their direct outputs/inputs
    output2norm, input2norm, norm_names = get_norm_layer_io(graph)

    # 2 find & update all possible outputs/inputs that with per-channel operations only
    # assume for a per-channel operation layer, it has only one input/output
    new_outputs = list(output2norm.keys())
    tensor_shape = {}
    while new_outputs:
        temp_outputs = new_outputs[:]
        for node in graph.nodes():
            node_str = get_node_str(node)
            # found new outputs
            matches = pass_layer_pattern.findall(node_str)
            if matches:
                input_names = parse_input_names(node_str)
                for input_name in input_names:
                    if input_name in temp_outputs:
                        output_name = parse_output_name(node_str)
                        if output_name not in tensor_shape:
                            output_shape = parse_output_shape(node_str)
                            tensor_shape[output_name] = output_shape

                        # check channel dim consistency for view operation
                        if view_pattern.match(matches[0]):
                            if tensor_shape[output_name][CHANNEL_DIM] == tensor_shape[input_name][CHANNEL_DIM]:
                                output2norm[output_name] = output2norm[input_name]
                                new_outputs.append(output_name)
                        # process normally
                        else:
                            output2norm[output_name] = output2norm[input_name]
                            new_outputs.append(output_name)
        new_outputs = new_outputs[len(temp_outputs):]

    new_inputs = list(input2norm.keys())
    while new_inputs:
        temp_inputs = new_inputs[:]
        for node in graph.nodes():
            node_str = get_node_str(node)
            # found new inputs
            matches = pass_layer_pattern.findall(node_str)
            if matches:
                output_name = parse_output_name(node_str)
                if output_name not in tensor_shape:
                    output_shape = parse_output_shape(node_str)
                    tensor_shape[output_name] = output_shape

                if output_name in temp_inputs:
                    input_name = parse_input_names(node_str)[0]

                    # check channel dim consistency for view operation
                    if view_pattern.match(matches[0]):
                        if tensor_shape[output_name][CHANNEL_DIM] == tensor_shape[input_name][CHANNEL_DIM]:
                            input2norm[input_name] = input2norm[output_name]
                            new_inputs.append(input_name)
                    # process normally
                    else:
                        input2norm[input_name] = input2norm[output_name]
                        new_inputs.append(input_name)
        new_inputs = new_inputs[len(temp_inputs):]

    # 3 identify layers need to be pruned
    succ_layers = {}    # succeeding layers
    prec_layers = {}    # preceding layers
    shortcut_source_names = []
    risky_layer_names = []
    risky_skip_connect_names = []
    for node in graph.nodes():
        node_str = get_node_str(node)
        if tensor_op_pattern.match(node_str):
            input_names = parse_input_names(node_str)
            output_name = parse_output_name(node_str)
            for input_name in input_names:
                if input_name in output2norm:
                    layer_name = parse_module_name(node_str)
                    source_layer_name = output2norm[input_name]
                    if prunable_layer_pattern.match(node_str):
                        # normalized output may be inputs to multiple layers
                        if source_layer_name in succ_layers:
                            succ_layers[source_layer_name].append(layer_name)
                        else:
                            succ_layers[source_layer_name] = [layer_name, ]
                    if not allowed_layer_pattern.match(node_str):
                        risky_layer_names.append(source_layer_name)
                    if add_op_pattern.match(node_str):
                        risky_skip_connect_names.append(source_layer_name)
                    if norm_layer_pattern.match(node_str):
                        if source_layer_name in norm_names:
                            shortcut_source_names.append(source_layer_name)

            if output_name in input2norm:
                layer_name = parse_module_name(node_str)
                source_layer_name = input2norm[output_name]

                if prunable_layer_pattern.match(node_str):
                    # support single input to normalization layer
                    prec_layers[source_layer_name] = [layer_name, ]
                if not allowed_layer_pattern.match(node_str):
                    # check for not allowed layers
                    risky_layer_names.append(source_layer_name)

                # make sure there are no branches in the path
                norm_inputs = reverse_search_dict(source_layer_name, input2norm)
                for norm_input in norm_inputs:
                    if input_count[norm_input] > 1:
                        risky_layer_names.append(source_layer_name)
                        break

    risky_layer_names = list(set(risky_layer_names))
    for risky_layer_name in risky_layer_names:
        # supposed to be safe because prunable BNs is the intersection of prec & succ
        #if risky_layer_name in succ_layers:
        #    succ_layers.pop(risky_layer_name)
        if risky_layer_name in prec_layers:
            prec_layers.pop(risky_layer_name)

    risky_skip_connect_names = list(set(risky_skip_connect_names))
    shortcut_source_names = list(set(shortcut_source_names))

    for very_risky_name in risky_skip_connect_names+shortcut_source_names:
        if very_risky_name in succ_layers:
            succ_layers.pop(very_risky_name)

    return prec_layers, succ_layers, norm_names


def get_norm_layer_names(model, input_shape):
    prec_layers, succ_layers, norm_layer_names = get_pruning_layers(model, input_shape)
    prunable_norm_layer_names = list(set(succ_layers) & set(prec_layers))
    maskable_norm_layer_names = list(set(succ_layers) - set(prec_layers))
    return prunable_norm_layer_names, maskable_norm_layer_names, norm_layer_names
