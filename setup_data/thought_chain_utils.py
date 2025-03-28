import re
from tqdm import tqdm

regexspace = re.compile(r"\ +", re.IGNORECASE)
regexlines = re.compile(r"(\n(\ )?)+", re.IGNORECASE)

def get_role_prompt():
    return "You are a helpful and expert writing assistant. You will be given a section of a story or screenplay from the perspective of {character}. Please answer the following questions about the given statements and their relationship with the snippet provided."

def get_icl_role_prompt():
    return "You are a helpful and expert writing assistant. Please answer the following questions to the best of your ability."

def get_section_text(section):
    template = """Story Section:    
{section}"""

    nice_section = section.strip()
    
    nice_section = regexspace.sub(" ", nice_section)
    nice_section = regexlines.sub("\n", nice_section)

    section_text = template.format(section=nice_section)
    return section_text

def get_final_entailment_question(prior_reasoning=False):
    if prior_reasoning:
        return """Taking your prior answers into account, rate the accuracy of provided statement about {character} on a scale of 1-5, where 1 is entirely inaccurate or unsupported and 5 is entirely accurate. If there are no claims made in the statement, mark the consistency of the statement as 1 as there is no evidence for the statement."""
    
    return """Rate the accuracy of provided statement about {character} on a scale of 1-5, where 1 is entirely inaccurate or unsupported and 5 is entirely accurate. If there are no claims made in the statement, mark the consistency of the statement as 1 as there is no evidence for the statement. Your response should be formatted as 'Answer: <number>'
Notes for accuracy:
Statements that are do not give us new information about {character}, or are not about {character} should be marked as 1. Even if the statement is true (e.g. 'X has no goals'), it should be marked as 1 as it does not give us new information about the character.
Statements that are not supported by the story snippet should be marked as 1.
Statements that are supported by the story snippet should be marked as 4 or 5, depending on how much evidence there is for the statement."""


def get_final_boolean_entailment_question(prior_reasoning=False):
    if prior_reasoning:
        return "Taking your prior answers into account, is the provided statement about {character} provably true using the snippet? Respond with \"Yes\" or \"No\", where \"No\" means the statement is entirely inaccurate or unsupported and \"Yes\" is entirely accurate and supported. If there are no claims made in the statement, mark the consistency of the statement as \"No\" as there is no evidence for the statement."
    else:
        return "Is the provided statement about {character} provably true using the snippet? Respond with \"Yes\" or \"No\", where \"No\" means the statement is entirely inaccurate or unsupported and \"Yes\" is entirely accurate and supported. If there are no claims made in the statement, mark the consistency of the statement as \"No\" as there is no evidence for the statement."
    
def get_entailment_questions():
    q1 = """What, if any, section of the story snippet is most relevant to the given statement? Provide a brief 1-2 sentence description of this section or "N/A" if there is no relevant section."""
    q2 = """In 1-2 sentences, compare the claim the statement makes and the section of story you highlighted in your previous answer. Are there any notable differences? Are all claims made by the statement explicitly supported? If there are no claims, write "N/A"."""
    return [q1, q2]

def get_prior_messages(qsandas):
    messages = []
    for q, a in qsandas:
        q_message = {"role": "user", "content": "Question: " + q}
        a_message = {"role": "assistant", "content": "Answer: " + a}
        messages.append(q_message)
        messages.append(a_message)
    return messages

def get_prior_messages_no_formatting(qsandas):
    messages = []
    for q, a in qsandas:
        q_message = {"role": "user", "content": q}
        a_message = {"role": "assistant", "content": a}
        messages.append(q_message)
        messages.append(a_message)
    return messages

def format_prompt(CHECKPOINT_NAME, TOKENIZER, character, story_section, statement, cur_question, prior_questions_and_answers):
    # llama has a separate section for roles
    using_system_role = "Llama" in CHECKPOINT_NAME
    
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_role_prompt().format(character=character)
    statement_text = statement.strip()
    story_section_text = get_section_text(story_section)
    # prior_question_text = get_prior_question_text(prior_questions_and_answers)
    prior_question_messages = get_prior_messages(prior_questions_and_answers)
    cur_question = cur_question.strip()
    user_message = None
    
    # add question message separately after question messages
    question_message = f"Question: {cur_question}".format(character=character)
    
    if using_system_role:
        prior_to_questions_template = """{story_section_text}

Please answer the following questions about {character} by comparing the provided statement with the story section above:

Statement: {statement_text}"""
        user_message = prior_to_questions_template.format(
                                    story_section_text=story_section_text, 
                                    statement_text=statement_text, 
                                    character=character)
    else:
        prior_to_questions_template = """{role}

{story_section_text}

Please answer the following questions about {character} by comparing the provided statement with the story section above:

Statement: {statement_text}"""
        user_message = prior_to_questions_template.format(role=role, 
                                    story_section_text=story_section_text, 
                                    statement_text=statement_text, 
                                    character=character)
    
    try:
        # some of the text might have the {character} formatting string that still needs to be infilled
        user_message = user_message.format(character=character)
    except:
        # this error arises when user_message contains some brackets from the story
        pass
    
    if len(prior_question_messages) == 0:
        user_message += "\n\n" + question_message
    
    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )
    
    # this and the if statement above are mutually exclusive
    # if there are prior messages then we add the question message separately
    # if there are not prior messages, then it's already in user_message
    # unfortunately we have to retcon the first question into the chat template
    if len(prior_question_messages) > 0:
        messages[-1]['content'] += "\n\n" + prior_question_messages[0]['content']
        messages.extend(prior_question_messages[1:])
        messages.append({"role":"user", "content":question_message})
    
    tokenized_chat = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokenized_chat += "Answer: "
    return tokenized_chat

def format_onlyreasoning_final_prompt(CHECKPOINT_NAME, TOKENIZER, character, statement, cur_question, prior_questions_and_answers):
    # llama has a separate section for roles
    using_system_role = "Llama" in CHECKPOINT_NAME
    
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_icl_role_prompt().format(character=character)
    statement_text = statement.strip()
    # prior_question_text = get_prior_question_text(prior_questions_and_answers)
    prior_question_messages = get_prior_messages(prior_questions_and_answers)
    cur_question = cur_question.strip()
    user_message = None
    
    # add question message separately after question messages
    question_message = f"Question: {cur_question}".format(character=character)
    
    if using_system_role:
        prior_to_questions_template = """Please answer the following questions about {character} and the provided statement.

Statement: {statement_text}"""
        user_message = prior_to_questions_template.format(
                                    statement_text=statement_text, 
                                    character=character)
    else:
        prior_to_questions_template = """{role}

Please answer the following questions about {character} and the provided statement.

Statement: {statement_text}"""
        user_message = prior_to_questions_template.format(role=role,
                                    statement_text=statement_text, 
                                    character=character)
    # some of the text might have the {character} formatting string that still needs to be infilled
    user_message.format(character=character)
    
    if len(prior_question_messages) == 0:
        user_message += "\n\n" + question_message
    
    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )
    
    # this and the if statement above are mutually exclusive
    # if there are prior messages then we add the question message separately
    # if there are not prior messages, then it's already in user_message
    # unfortunately we have to retcon the first question into the chat template
    if len(prior_question_messages) > 0:
        messages[-1]['content'] += "\n\n" + prior_question_messages[0]['content']
        messages.extend(prior_question_messages[1:])
        messages.append({"role":"user", "content":question_message})
    
    tokenized_chat = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokenized_chat += "Answer: "
    return tokenized_chat

def format_icl_prompt_multistep(CHECKPOINT_NAME, TOKENIZER, character, statement, cur_question, prior_questions_and_answers, answer_prefix="Answer: "):
    # llama has a separate section for roles
    using_system_role = "Llama" in CHECKPOINT_NAME
    
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_icl_role_prompt().format(character=character)
    statement_text = statement.strip()
    # prior_question_text = get_prior_question_text(prior_questions_and_answers)
    prior_question_messages = get_prior_messages_no_formatting(prior_questions_and_answers)
    cur_question = cur_question.strip()
    user_message = None
    
    # add question message separately after question messages
    question_message = f"Question: {cur_question}\nStatement: {statement_text}".format(character=character)
    user_message = ""
    if not using_system_role:
        user_message = role
    
    # some of the text might have the {character} formatting string that still needs to be infilled
    user_message.format(character=character)
    
    if len(prior_question_messages) == 0:
        user_message += "\n\n" + question_message
    
    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )
    
    # this and the if statement above are mutually exclusive
    # if there are prior messages then we add the question message separately
    # if there are not prior messages, then it's already in user_message
    # unfortunately we have to retcon the first question into the chat template
    # print(prior_question_messages[:2])
    if len(prior_question_messages) > 0:
        messages[-1]['content'] += "\n\n" + prior_question_messages[0]['content']
        messages.extend(prior_question_messages[1:])
        messages.append({"role":"user", "content":question_message})
    
    messages.append({"role":"assistant", "content":answer_prefix})
    
    tokenized_chat = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    tokenized_chat = tokenized_chat[:,:-1]
    
    return tokenized_chat

def format_icl_prompt(CHECKPOINT_NAME, TOKENIZER, character, statement, cur_question, prior_questions_and_answers):
    # llama has a separate section for roles
    using_system_role = "Llama" in CHECKPOINT_NAME
    
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_icl_role_prompt().format(character=character)
    statement_text = statement.strip()
    # prior_question_text = get_prior_question_text(prior_questions_and_answers)
    prior_question_messages = get_prior_messages(prior_questions_and_answers)
    cur_question = cur_question.strip()
    user_message = None
    
    # add question message separately after question messages
    question_message = f"Question: {cur_question}\nStatement: {statement_text}".format(character=character)
    user_message = ""
    if not using_system_role:
        user_message = role
    
    # some of the text might have the {character} formatting string that still needs to be infilled
    user_message.format(character=character)
    
    if len(prior_question_messages) == 0:
        user_message += "\n\n" + question_message
    
    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )
    
    # this and the if statement above are mutually exclusive
    # if there are prior messages then we add the question message separately
    # if there are not prior messages, then it's already in user_message
    # unfortunately we have to retcon the first question into the chat template
    if len(prior_question_messages) > 0:
        messages[-1]['content'] += "\n\n" + prior_question_messages[0]['content']
        messages.extend(prior_question_messages[1:])
        messages.append({"role":"user", "content":question_message})
    
    tokenized_chat = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    return tokenized_chat
    
def format_icl_prompt_with_answer(CHECKPOINT_NAME, TOKENIZER, character, statement, cur_question, answer_text):
    # llama has a separate section for roles
    using_system_role = "Llama" in CHECKPOINT_NAME
    
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_icl_role_prompt().format(character=character)
    statement_text = statement.strip()
    cur_question = cur_question.strip()
    user_message = None
    
    # add question message separately after question messages
    question_message = f"Question: {cur_question}\nStatement: {statement_text}".format(character=character)
    user_message = ""
    if not using_system_role:
        user_message = role
    
    # some of the text might have the {character} formatting string that still needs to be infilled
    user_message.format(character=character)
    
    user_message += "\n\n" + question_message
    
    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )
    messages.append(
        {"role": "assistant", "content": answer_text},
    )
    
    tokenized_chat = TOKENIZER.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return tokenized_chat
    
def format_single_qa_prompt(CHECKPOINT_NAME, TOKENIZER, character, statement, cur_question, prior_questions_and_answers):
    # llama has a separate section for roles
    using_system_role = "Llama" in CHECKPOINT_NAME
    
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_icl_role_prompt().format(character=character)
    statement_text = statement.strip()
    # prior_question_text = get_prior_question_text(prior_questions_and_answers)
    prior_question_messages = get_prior_messages(prior_questions_and_answers)
    cur_question = cur_question.strip()
    user_message = None
    
    # add question message separately after question messages
    question_message = f"Question: {cur_question}\nStatement: {statement_text}".format(character=character)
    user_message = ""
    if not using_system_role:
        user_message = role
    
    # some of the text might have the {character} formatting string that still needs to be infilled
    user_message.format(character=character)
    
    if len(prior_question_messages) == 0:
        user_message += "\n\n" + question_message
    
    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )
    
    # this and the if statement above are mutually exclusive
    # if there are prior messages then we add the question message separately
    # if there are not prior messages, then it's already in user_message
    # unfortunately we have to retcon the first question into the chat template
    if len(prior_question_messages) > 0:
        messages[-1]['content'] += "\n\n" + prior_question_messages[0]['content']
        messages.extend(prior_question_messages[1:])
        messages.append({"role":"user", "content":question_message})
    
    tokenized_chat = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenized_chat

def get_tokens_as_list(word_list, TOKENIZER):
    "Converts a sequence of words into a list of tokens"
    # huggingface util (https://huggingface.co/docs/transformers/internal/generation_utils) for 
    # getting bad_words_list
    tokens_list = []
    for word in word_list:
        tokenized_word = TOKENIZER([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

def run_list_of_prompts_through_model(queries, pipe, MAX_TOKENS, NUM_BEAMS, bad_words_ids, TOKENIZER):
    print(f"working on queries: {len(queries)}")
    text_per_prompt = []
    sequences = pipe(
        queries,
        return_full_text=False,
        do_sample=False,
        max_new_tokens=MAX_TOKENS, 
        num_beams=NUM_BEAMS, 
        num_return_sequences=1,
        bad_words_ids=bad_words_ids
    )
    text_per_prompt = [resp[0]['generated_text'] for resp in sequences]
    
    return text_per_prompt

def icl_run_list_of_prompts_through_model(queries, model, MAX_TOKENS, NUM_BEAMS, bad_words_ids, TOKENIZER):
    print(f"working on queries: {len(queries)}")
    text_per_prompt = []
    for inputs in tqdm(queries):
        output = model.generate(
            inputs.to(model.device),
            do_sample=False,
            max_new_tokens=MAX_TOKENS, 
            num_beams=NUM_BEAMS, 
            num_return_sequences=1,
            bad_words_ids=bad_words_ids
        )
        decoded = TOKENIZER.decode(output[0])

        text_per_prompt.append(decoded)
    
    return text_per_prompt