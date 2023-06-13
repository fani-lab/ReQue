from transformers import pipeline

import sys
sys.path.extend(['../qe'])

from expanders.abstractqexpander import AbstractQExpander


class BackTranslation(AbstractQExpander):
    def __init__(self):
        AbstractQExpander.__init__(self)

        #TODO add a constant file - check lady project

        # Constanrs
        src = 'eng_Latn'
        tgt = 'fra_Latn'
        max_length = 512
        device = 'cpu'
        nllb = 'facebook/nllb-200-distilled-600M'

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(nllb)
        tokenizer = AutoTokenizer.from_pretrained(nllb)

        self.translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src, tgt_lang=tgt,
                                   max_length=max_length, device=device)
        self.back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=tgt, tgt_lang=src,
                                        max_length=max_length, device=device)

    def get_expanded_query(self, q, args=None):
        translated_query = self.translator(q)
        back_translated_query = self.back_translator(translated_query[0]['translation_text'])

        return back_translated_query[0]['translation_text']


if __name__ == "__main__":
    qe = BackTranslation()
    print(qe.get_model_name())
    print(qe.get_expanded_query('This is my pc'))

