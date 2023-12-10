# In MNLI datasets, labels are 0 and 2 (1 is deleted). In HANS dataset, labels are 0 and 1
minimal_pattern_labels_map = {
    0: "Yes",
    1: "No",
    2: "No"
}
eval_harness_labels_map = {
    0: "True",
    1: "False",
    2: "False"
}

def minimal_pattern_creator_5(premise, hypothesis, label):
    if len(label) > 0:
        label = " " + label + "."
    return f"{premise} {hypothesis}? {label}"

def gpt3_pattern_creator(premise, hypothesis, label):
    if len(label) > 0:
        label = " " + label + "."
    return f"{premise} question: {hypothesis} Yes or No? answer:{label}"

def eval_harness_pattern_creator_2(premise, hypothesis, label):
    if len(label) > 0:
        label = " " + label + "."
    return f"{premise} \n Question: {hypothesis} True or False? answer:{label}"

patterns_dict = {
    "minimal": {
        "creator": minimal_pattern_creator_5,
        "labels_map": minimal_pattern_labels_map
    },
    "gpt-3": {
        "creator": gpt3_pattern_creator,
        "labels_map": minimal_pattern_labels_map
    },
    "eval-harness": {
        "creator": eval_harness_pattern_creator_2,
        "labels_map": eval_harness_labels_map
    }
}