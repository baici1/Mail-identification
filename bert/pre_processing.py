from transformers import BertTokenizer


def pre_processing(mail, pre_train_model="chinese-bert-wwm", max_length=512):
    """return Bert input

    Args:
        mail (str): the mail text

    Returns:
        dir {str:tensor}: the input of BertModel
    """
    tokenizer = BertTokenizer.from_pretrained(pre_train_model)
    mail_token = tokenizer(
        mail,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return mail_token
