from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import sys

sys.path.extend(['../qe'])

from expanders.abstractqexpander import AbstractQExpander
from cmn import param


class BackTranslation(AbstractQExpander):
    def __init__(self, tgt):
        AbstractQExpander.__init__(self)

        self.tgt = tgt

        model = AutoModelForSeq2SeqLM.from_pretrained(param.backtranslation['model_card'])
        tokenizer = AutoTokenizer.from_pretrained(param.backtranslation['model_card'])

        self.transformer_model = SentenceTransformer(param.backtranslation['transformer_model'])

        self.translator = pipeline("translation", model=model, tokenizer=tokenizer,
                                   src_lang=param.backtranslation['src_lng'], tgt_lang=self.tgt,
                                   max_length=param.backtranslation['max_length'],
                                   device=param.backtranslation['device'])
        self.back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=self.tgt,
                                        tgt_lang=param.backtranslation['src_lng'],
                                        max_length=param.backtranslation['max_length'],
                                        device=param.backtranslation['device'])

    def get_expanded_query(self, q, args=None):
        translated_query = self.translator(q)
        back_translated_query = self.back_translator(translated_query[0]['translation_text'])

        score = self.semsim(q, back_translated_query[0]['translation_text'])
        return super().get_expanded_query(back_translated_query[0]['translation_text'], [score])

        # score = self.semsim(q, q)
        # return super().get_expanded_query(q, [score])

    def get_model_name(self):
        return super().get_model_name() + '_' + self.tgt.lower()

    def semsim(self, q1, q2):
        me, you = self.transformer_model.encode([q1, q2])
        return 1 - cosine(me, you)


if __name__ == "__main__":
    qe = BackTranslation()
    print(qe.get_model_name())
    print(qe.get_expanded_query('This is my pc'))
