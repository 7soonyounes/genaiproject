import sacrebleu
from rouge_score import rouge_scorer

class PoetryComparison:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_bleu(self, reference, candidate):
        bleu = sacrebleu.corpus_bleu(candidate, [reference])
        return bleu.score

    def calculate_rouge(self, reference, candidate):
        scores = self.rouge_scorer.score(reference, candidate)
        return scores

    def compare_poems(self, gpt2_poem, gpt_neo_poem):
        bleu_score = self.calculate_bleu(gpt2_poem, gpt_neo_poem)
        rouge_scores = self.calculate_rouge(gpt2_poem, gpt_neo_poem)
        
        return {
            "BLEU": bleu_score,
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-2": rouge_scores["rouge2"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure
        }