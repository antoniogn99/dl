from transformers import AutoTokenizer, OPTForCausalLM


def load_model(model_name):
    if model_name == "opt-125m":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", padding_side='left')
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
    elif model_name == "opt-350m":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", padding_side='left')
        model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
    elif model_name == "opt-1.3b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side='left')
        model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
    else:
        raise ValueError("Model name is not valid")
    return tokenizer, model

def query_model(model_name, input_texts, possible_labels):
    tokenizer, model = load_model(model_name)
    positive_label = possible_labels[0]
    negative_label = possible_labels[1]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1, do_sample=True)
    generated_token_scores = generation_output.scores[-1]
    positive_token_id = tokenizer.encode(positive_label, return_tensors="pt")[0][1].item()
    negative_token_id = tokenizer.encode(negative_label, return_tensors="pt")[0][1].item()
    scores = generated_token_scores[:, [positive_token_id, negative_token_id]]
    temp_dict = {
        0: positive_label,
        1: negative_label
    }
    return [temp_dict[int(x.item())] for x in scores.argmax(axis=1)]


if __name__ == "__main__":
    input_text_1 = "Review: Delicious food! Sentiment: Positive. The food is awful. Sentiment: Negative. Review: Terrible dishes. Sentiment: Negative. Review: Bad meal. Sentiment:"
    input_text_2 = "Review: Great meal! Sentiment: Positive. Review: Delicious food! Sentiment: Positive. Review: Very good food! Sentiment: Positive. Review: Excellent meal. Sentiment:"
    preds = query_model("opt-125m", [input_text_1, input_text_2, input_text_1], ["Positive", "Negative"])
    print(preds)
    input_text_3 = "you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him. You lose the things to the following level if the people recall? Yes. One of our number will carry out your instructions minutely. A member of my team will execute your orders with immense precision? Yes. It's a commitment to general education--a sequence of courses intended to develop critical thinking in a wide variety of disciplines--in opposition to early specialization. General education's focus is to develop students' critical thinking skills?"
    input_text_4 = "you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him. You lose the things to the following level if the people recall? Yes. One of our number will carry out your instructions minutely. A member of my team will execute your orders with immense precision? Yes. yeah it's strange because well it it's not strange because i use to be the same way and i'm even to this day  you know some vegetables really turn me off but when you read so much information that says this is a healthier way to go you know and this is what your body wants this is what your body really needs and when you think about what is what's the real reason your eating i know i know it's for taste because i'm boy am i a taste person but. i'm a taste person and all vegetables taste very nice unlike other types of food?"
    preds = query_model("opt-125m", [input_text_3, input_text_4], ["Yes", "No"])
    print(preds)
