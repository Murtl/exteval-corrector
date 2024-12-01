import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from nltk import sent_tokenize
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
"https://storage.googleapis.com/allennlp-public-models/stanford-sentiment-treebank-roberta.2021-03-11.tar.gz",
 cuda_device=-1) # change to cuda_device=0 if you have a GPU and you got all the necessary libraries installed

def preprocess(document, summary):
    all_document_sents = document.lower().replace('"', "").replace("`", "").replace("''", "").replace('-', '').replace(
        "p ! nk", "p!nk").replace("d.j .", "d.j.").replace("d.c .", "d.c.").replace("u.s.", "u.s . ").replace(
        'p.m.', 'p.m .').replace('a.m.', 'a.m .').split("<br>")
    document_sents = [(di, sent.split("<u>")[1].split("</u>")[0].strip().lower().split())
                      for di, sent in enumerate(all_document_sents) if "<u>" in sent]
    summary = summary.lower().replace('"', "").replace("`", "").replace("-lrb-", "(").replace("-rrb-", ")").replace(
        "p ! nk", "p!nk").replace("''", "").replace("d.j .", "d.j.").replace("d.c .", "d.c.").replace(
        "u.s .", "u.s.").replace('-', '').replace("nyong  o", "nyong'o").replace("nyong ' o", "nyong'o").replace(
        'co .', 'co.').replace('mass.', 'mass').replace('dr.', 'dr .').replace('mexico.', 'mexico .').replace(
        "u.s.", "u.s . ").replace('a&m', 'a & m').replace('f * * * * * g', 'f*****g').replace(
        'p.m.', 'p.m .').replace('a.m.', 'a.m .')
    summary_sents = [sent.split('.', 1)[1].strip().split()
                     for sent in summary.split("<br>")]
    return all_document_sents, document_sents, summary_sents


def locate_summaries(document_sents, summary_sents):
    # locate summary sents
    locations = []
    summary_sent_indexes = []
    for senti, swords in enumerate(summary_sents):
        if len(swords) == 0:
            continue
        dptr, match = 0, 0
        while dptr < max(map(lambda x: len(x[1]), document_sents)):
            for di, dwords in document_sents:
                if dptr >= len(dwords):
                    continue
                sptr_tmp, dptr_tmp = 0, dptr
                while sptr_tmp < len(swords) and dptr_tmp < len(dwords):
                    dword = dwords[dptr_tmp]
                    sword = swords[sptr_tmp]
                    if "[" in dword or "<span" in dword or "span>" in dword or "color:#" in dword:
                        dptr_tmp += 1
                        continue
                    elif "[" in sword or "<span" in sword or "span>" in sword or "color:#" in sword:
                        sptr_tmp += 1
                        continue
                    elif dword == sword:
                        match += 1
                    else:
                        break
                    sptr_tmp += 1
                    dptr_tmp += 1
                    if match >= 5:
                        break
                if match >= 5 or match / len(swords) > 0.9:
                    locations.append([di, dptr])
                    summary_sent_indexes.append(senti)
                    break
                else:
                    match = 0
            if match >= 5 or match / len(swords) > 0.9:
                break
            dptr += 1
    return locations, summary_sent_indexes

def coref_disco_metric(document, summary):
    # get document and summary sentences
    all_document_sents, document_sents, summary_sents = preprocess(document, summary)

    # locate summary sentences in document
    locations, summary_sent_indexes = locate_summaries(document_sents, summary_sents)

    errors = {
        "IncorCorefEval": {"count": 0, "details": []},  # Incorrect Coreference
        "IncomCorefEval": {"count": 0, "details": []},  # Incomplete Coreference
        "IncomDiscoEval": {"count": 0, "details": []},  # Incomplete Discourse
    }

    # match coreference
    scorefs, dscorefs, scorefs_map, rev_scorefs_map = {}, {}, {}, {}
    scoref, smention, dcoref, dmention = None, [], None, []

    for si, senti in enumerate(summary_sent_indexes):
        swords = summary_sents[senti]
        di, dptr = locations[si]
        dwords = dict(document_sents)[di]

        sptr, plain_words = 0, []
        while sptr < len(swords) and dptr < len(dwords):
            sword = swords[sptr]
            dword = dwords[dptr]

            if "<span" in sword or "color:#" in sword:
                # if <span or color in sword, then go to next word in the summary
                sptr += 1
            elif "[" in sword:
                # if coref number in sword, save it to scoref and go to next summary word
                scoref = sword
                sptr += 1
            elif "span>" in sword:
                # if it is the end of a span, save scoref and smention
                if scoref not in scorefs:
                    scorefs[scoref] = []
                scorefs[scoref].append(smention)

                if dcoref is not None and "span>" in dword and dmention == smention:
                    # if there is a same mention in document
                    if scoref not in scorefs_map:
                        scorefs_map[scoref] = dcoref  # save the map
                        rev_scorefs_map[dcoref] = scoref  # save the reverse map
                    else:
                        # IncorCoref: if the mapping contradicts previously saved mapping
                        if scorefs_map[scoref] != dcoref:
                            errors["IncorCorefEval"]["count"] += 1
                            errors["IncorCorefEval"]["details"].append({
                                "summary_sentence": senti,
                                "mention": smention,
                                "conflicting_mappings": (scorefs_map[scoref], dcoref)
                            })

                    dcoref, dmention = None, []
                    dptr += 1

                # if it is the first mention of scoref
                if len(scorefs[scoref]) == 1 and di != 0:
                    # IncomCoref
                    if len(smention) > 1 and smention[0] in ["the", "that", "this", "these", "those", "both"]:
                        if scoref in scorefs_map:
                            # check if there is antecedent in the document
                            exist_antecedent = False
                            for doc_sent in all_document_sents[:di]:
                                if scorefs_map[scoref] in doc_sent:
                                    exist_antecedent = True
                                    break
                            # there is an antecedent in the document, and the summary fails to include it
                            if exist_antecedent:
                                errors["IncomCorefEval"]["count"] += 1
                                errors["IncomCorefEval"]["details"].append({
                                    "summary_sentence": senti,
                                    "mention": smention,
                                    "missing_antecedent": scorefs_map[scoref]
                                })

                    # IncomCoref (only when mention length=1)
                    if len(smention) == 1 and smention[0] in ["he", "she", "him", "her", "his", "they",
                                                              "them", "their", "it", "this", "that", "those",
                                                              "these"]:
                        errors["IncomCorefEval"]["count"] += 1
                        errors["IncomCorefEval"]["details"].append({
                            "summary_sentence": senti,
                            "mention": smention,
                            "type": "ambiguous_pronoun"
                        })

                scoref, smention = None, []
                sptr += 1

            else:
                plain_words.append(sword)
                # if it is a normal word
                if scoref is not None:
                    smention.append(sword)  # update smention if there is a scoref

                # find the corresponding word in the document
                if dword != sword:
                    while dword != sword:
                        if "[" in dword:
                            dcoref = dword
                        elif dcoref is not None and "span>" in dword:
                            if dcoref in rev_scorefs_map:  # if this dcoref can be mapped to a previous scoref
                                scorefs[rev_scorefs_map[dcoref]].append(dmention)
                            else:
                                if dcoref not in dscorefs:  # save dcoref to dscorefs
                                    dscorefs[dcoref] = []
                                dscorefs[dcoref].append(dmention)

                                # if it is the first mention of dcoref in summary
                                if len(dscorefs[dcoref]) == 1 and di != 0:
                                    # IncomCoref
                                    if len(dmention) > 1 and dmention[0] in ["the", "that", "this", "these",
                                                                             "those", "both"]:
                                        # check if there is antecedent in the document
                                        exist_antecedent = False
                                        for doc_sent in all_document_sents[:di]:
                                            if dcoref in doc_sent:
                                                exist_antecedent = True
                                                break
                                        # there is an antecedent in the document, and the summary fails to include it
                                        if exist_antecedent:
                                            errors["IncomCorefEval"]["count"] += 1
                                            errors["IncomCorefEval"]["details"].append({
                                                "summary_sentence": senti,
                                                "mention": dmention,
                                                "missing_antecedent": dcoref
                                            })

                                    # IncomCoref (only when mention length=1)
                                    if len(dmention) == 1 and dmention[0] in ["he", "she", "him", "her", "his", "they",
                                                              "them", "their", "it", "this", "that", "those", "these"]:
                                        errors["IncomCorefEval"]["count"] += 1
                                        errors["IncomCorefEval"]["details"].append({
                                            "summary_sentence": senti,
                                            "mention": dmention,
                                            "missing_antecedent": dcoref
                                        })
                            dcoref, dmention = None, []
                        dptr += 1
                        if dptr >= len(dwords):
                            break
                        try:
                            dword = dwords[dptr]
                        except:
                            print(dwords)
                            print(swords)
                            print(sword)
                            exit()

                if dcoref is not None:
                    dmention.append(dword)  # if there is a dcoref, then update dmention
                dptr += 1
                sptr += 1

        # IncomDisco
        if locations[si][1] == 0:  # it starts from the beginning of a sentence
            if plain_words[0] in ["and", "so", "still"]:
                if (si == 0 and di != 0) or locations[si - 1][0] != di - 1:
                    errors["IncomDiscoEval"]["count"] += 1
                    errors["IncomDiscoEval"]["details"].append({
                        "summary_sentence": senti,
                        "discourse_marker": plain_words[0]
                    })
            else:
                last_discourse_marker = None
                for key in ["also", "however", "but", "clearly", "meanwhile", "not only", "not just",
                            "on another", "then", "moreover"]:
                    if key in ' '.join(plain_words[:5]):
                        if (si == 0 and di != 0) or locations[si - 1][0] != di - 1:
                            errors["IncomDiscoEval"]["count"] += 1
                            errors["IncomDiscoEval"]["details"].append({
                                "summary_sentence": senti,
                                "discourse_marker": key
                            })
                if "on one" in ' '.join(plain_words[:5]):
                    if si == len(locations) - 1 or locations[si + 1][0] != di + 1:
                        errors["IncomDiscoEval"]["count"] += 1
                        errors["IncomDiscoEval"]["details"].append({
                            "summary_sentence": senti,
                            "discourse_marker": "on one"
                        })
        else:  # starts from the middle of a sentence
            if (si == 0 and di != 0) or locations[si - 1][0] != di:
                errors["IncomDiscoEval"]["count"] += 1
                errors["IncomDiscoEval"]["details"].append({
                    "summary_sentence": senti,
                    "discourse_marker": plain_words[0]
                })

    return errors


def get_sentiment_details(document, summary, batch_size=32):
    # Tokenize sentences for sentiment analysis
    doc_sents = [{"sentence": sent.strip()} for sent in sent_tokenize(document) if sent.strip()]
    summary_sents = [{"sentence": sent.strip()} for sent in sent_tokenize(summary) if sent.strip()]

    # Analyze document sentiment
    doc_sentiments = []
    for j in range(len(doc_sents) // batch_size + 1):
        batch = doc_sents[j * batch_size:(j + 1) * batch_size]
        if batch:
            results = predictor.predict_batch_json(batch)
            doc_sentiments.extend([result["probs"][0] for result in results])  # Positive sentiment probability

    # Analyze summary sentiment
    summary_sentiments = []
    for j in range(len(summary_sents) // batch_size + 1):
        batch = summary_sents[j * batch_size:(j + 1) * batch_size]
        if batch:
            results = predictor.predict_batch_json(batch)
            summary_sentiments.extend([result["probs"][0] for result in results])

    # Calculate detailed SentiBias
    doc_avg_sentiment = np.mean(doc_sentiments) if doc_sentiments else 0.5
    summary_avg_sentiment = np.mean(summary_sentiments) if summary_sentiments else 0.5
    absolute_difference = abs(doc_avg_sentiment - summary_avg_sentiment)

    # Identify specific deviations at the sentence level
    significant_deviations = [
        {
            "sentence_index": i,
            "summary_sentiment": summary_sentiments[i],
            "document_avg_sentiment": doc_avg_sentiment
        }
        for i in range(len(summary_sentiments))
        if abs(summary_sentiments[i] - doc_avg_sentiment) > 0.2  # Threshold for significant deviation
    ]

    return {
        "doc_avg_sentiment": doc_avg_sentiment,
        "summary_avg_sentiment": summary_avg_sentiment,
        "absolute_difference": absolute_difference,
        "significant_deviations": significant_deviations
    }


def exteval(data, batch_size=32):
    all_exteval = {}
    for key in tqdm(data):
        example = data[key]
        assert "document_for_annotation" in example and "corrected_extractive_summary_for_annotation" in example
        document_for_annotation = example["document_for_annotation"]
        summary_for_annotation = example["corrected_extractive_summary_for_annotation"]

        # Get coreference and discourse errors
        coref_disco_results = coref_disco_metric(document_for_annotation, summary_for_annotation)

        # Calculate detailed sentiment bias
        document = example["document"]
        summary = example["corrected_extractive_summary"].replace("<t>", "").replace("</t>", " ")
        sentiment_details = get_sentiment_details(document, summary, batch_size=batch_size)

        # Combine all metrics
        results = {
            "IncorCorefEval": coref_disco_results["IncorCorefEval"],  # Includes count and details
            "IncomCorefEval": coref_disco_results["IncomCorefEval"],  # Includes count and details
            "IncomDiscoEval": coref_disco_results["IncomDiscoEval"],  # Includes count and details
            "SentiBias": {
                "absolute_difference": sentiment_details["absolute_difference"],
                "doc_avg_sentiment": sentiment_details["doc_avg_sentiment"],
                "summary_avg_sentiment": sentiment_details["summary_avg_sentiment"],
                "significant_deviations": sentiment_details["significant_deviations"]
            }
        }

        # Calculate the overall EXTEVAL score
        exteval_score = (
            (1 if coref_disco_results["IncorCorefEval"]["count"] > 0 else 0) +
            (1 if coref_disco_results["IncomCorefEval"]["count"] > 0 else 0) +
            (1 if coref_disco_results["IncomDiscoEval"]["count"] > 0 else 0) +
            sentiment_details["absolute_difference"]
        )
        results["ExtEval"] = exteval_score

        # Store results for this example
        all_exteval[key] = results

    return all_exteval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=None,
                        type=str, help="The input data file (in json format).")
    parser.add_argument("--output_file", default=None,
                        type=str, help="The output file")
    parser.add_argument("--batch_size", default=32, type=int, help="Eval batch size")

    args = parser.parse_args()
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    start_time = time.time()
    all_exteval = exteval(data, batch_size=args.batch_size)
    end_time = time.time()
    print(end_time - start_time)
    with open(args.output_file, 'w') as f:
        json.dump(all_exteval, f, indent=4)