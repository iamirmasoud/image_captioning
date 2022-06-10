from nltk.translate.bleu_score import corpus_bleu


def clean_sentence(output, idx2word):
    sentence = ""
    for i in output:
        word = idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentence = sentence + word
        else:
            sentence = sentence + " " + word
    return sentence


def bleu_score(true_sentences, predicted_sentences):
    hypotheses = []
    references = []
    for img_id in set(true_sentences.keys()).intersection(
        set(predicted_sentences.keys())
    ):
        img_refs = [cap.split() for cap in true_sentences[img_id]]
        references.append(img_refs)
        hypotheses.append(predicted_sentences[img_id][0].strip().split())

    return corpus_bleu(references, hypotheses)
