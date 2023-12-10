from datasets import load_dataset, Dataset
from patterns import patterns_dict
from models import query_model
import matplotlib.pyplot as plt
import numpy as np


def compute_accuracy(demonstrations_dataset, evaluation_dataset, model_name="opt-125m", pattern="minimal", demons_per_query=16):
    pattern_labels_map = patterns_dict[pattern]["labels_map"]

    # Add context
    def create_context():
        row = np.random.randint(0, len(demonstrations_dataset)-demons_per_query)
        temp_dataset = demonstrations_dataset.select([i for i in range(row, row + demons_per_query)])
        return ' '.join(temp_dataset[f"text_{pattern}"])
    evaluation_dataset = evaluation_dataset.map(
        lambda example: {"context": create_context()}
    )
    
    # Put together context and premise-hypothesis pattern to create input text
    evaluation_dataset = evaluation_dataset.map(
        lambda example: {"input": 
            ' '.join([example["context"], example[f"text_{pattern}"]])
        }
    )

    # Use input text to query model
    possible_labels = list(pattern_labels_map.values())
    pred_labels = query_model(model_name, evaluation_dataset["input"], possible_labels)

    # Compare with true kabels
    true_labels = [pattern_labels_map[x] for x in evaluation_dataset["label"]]
    correct = 0
    incorrect = 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label==pred_label:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct/(correct+incorrect)
    return accuracy

def main():
    # Read data
    mnli_dataset = load_dataset("glue", "mnli")
    hans_dataset = load_dataset("hans")
    demonstrations_dataset = mnli_dataset["train"].filter(lambda example: example["label"] != 1)
    in_domain_evaluation_dataset = mnli_dataset["validation_matched"].filter(lambda example: example["label"] != 1)
    out_of_domain_evaluation_dataset = hans_dataset["validation"]

    # Shuffle data
    demonstrations_dataset = demonstrations_dataset.shuffle()
    in_domain_evaluation_dataset = in_domain_evaluation_dataset.shuffle()
    out_of_domain_evaluation_dataset = out_of_domain_evaluation_dataset.shuffle()

    # Add columns with the patterns
    patterns_list = ["minimal", "gpt-3", "eval-harness"]
    for pattern in patterns_list:
        pattern_creator = patterns_dict[pattern]["creator"]
        pattern_labels_map = patterns_dict[pattern]["labels_map"]

        # Create premise-hypothesis-label pattern in the demonstrations dataset
        demonstrations_dataset = demonstrations_dataset.map(
            lambda example: {f"text_{pattern}": pattern_creator(
                premise = example["premise"],
                hypothesis = example["hypothesis"], 
                label = pattern_labels_map[example["label"]]
            )}
        )
    
        #  Create premise-hypothesis pattern (label not included) in the evaluation datasets
        in_domain_evaluation_dataset = in_domain_evaluation_dataset.map(
            lambda example: {f"text_{pattern}": pattern_creator(
                premise = example["premise"],
                hypothesis = example["hypothesis"], 
                label = ""
            )}
        )
        out_of_domain_evaluation_dataset = out_of_domain_evaluation_dataset.map(
            lambda example: {f"text_{pattern}": pattern_creator(
                premise = example["premise"],
                hypothesis = example["hypothesis"], 
                label = ""
            )}
        )

    # This can be increased if the computer is good enough. It should be 16
    demons_per_query = 2

    # Iterate through all  models and patterns
    models_list = ["opt-125m", "opt-350m", "opt-1.3b"]
    patterns_list = ["minimal", "gpt-3", "eval-harness"]

    accuracy_dict = {}
    num_batches = 10
    batch_size = 1000
    for model_name in models_list:
        print("Model:", model_name)
        for pattern in patterns_list:
            print("* Pattern:", pattern)
            accuracy_dict[(model_name, pattern)] = []
            for batch_index in range(num_batches):
                in_domain_acc = compute_accuracy(demonstrations_dataset, in_domain_evaluation_dataset.shuffle().select([i for i in range(batch_size)]), model_name, pattern, demons_per_query)
                out_of_domain_acc = compute_accuracy(demonstrations_dataset, out_of_domain_evaluation_dataset.shuffle().select([i for i in range(batch_size)]), model_name, pattern, demons_per_query)
                accuracy_dict[(model_name, pattern)].append((in_domain_acc, out_of_domain_acc))
                print(f"--- Accuracy in batch {batch_index}: {in_domain_acc}, {out_of_domain_acc}")

    print(accuracy_dict)
    plot_accuracies(accuracy_dict, models_list, patterns_list)

def plot_accuracies(accuracy_dict, models_list, patterns_list):
    for pattern in patterns_list:
        for model in models_list:
            x = [in_domain_acc for in_domain_acc, out_of_domain_acc in accuracy_dict[(model, pattern)]]
            y = [out_of_domain_acc for in_domain_acc, out_of_domain_acc in accuracy_dict[(model, pattern)]]
            plt.plot(x, y, "X", label=model, alpha=0.5, markersize=10, markeredgewidth=0.0)
        plt.xlabel("in-domain accuracy")
        plt.xlim([0.4, 0.9])
        plt.ylabel("out-of-domain accuracy")
        plt.ylim([0.4, 0.9])
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(pattern)
        plt.savefig(f"icl-{pattern}.png")
        plt.clf()

if __name__ == "__main__":
    main()
