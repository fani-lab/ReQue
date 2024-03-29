from cmn import param
from expanders.abstractqexpander import AbstractQExpander
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import sys

sys.path.extend(['../qe'])


class BackTranslation(AbstractQExpander):
    def __init__(self, tgt):
        AbstractQExpander.__init__(self)

        # Initialization
        self.tgt = tgt
        model = AutoModelForSeq2SeqLM.from_pretrained(
            param.backtranslation['model_card'])
        tokenizer = AutoTokenizer.from_pretrained(
            param.backtranslation['model_card'])

        # Translation models
        self.translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=param.backtranslation[
                                   'src_lng'], tgt_lang=self.tgt, max_length=param.backtranslation['max_length'], device=param.backtranslation['device'])
        self.back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=self.tgt,
                                        tgt_lang=param.backtranslation['src_lng'], max_length=param.backtranslation['max_length'], device=param.backtranslation['device'])
        # Model use for calculating semsim
        self.transformer_model = SentenceTransformer(
            param.backtranslation['transformer_model'])

    # Generate the backtranslated of the original query then calculates the difference of the two queries
    def get_expanded_query(self, q, args=None):
        translated_query = self.translator(q)
        back_translated_query = self.back_translator(
            translated_query[0]['translation_text'])

        with open('output\\robust04\\translatedqueries.txt', 'a+') as outfile:
            # output qid, original query, translated query, backtranslated query
            outfile.write(str(args[0]) + '\t' + str(q) + '\t' + translated_query[0]['translation_text'] + '\t' + back_translated_query[0]['translation_text'] + '\n')

        score = self.semsim(q, back_translated_query[0]['translation_text'])
        return super().get_expanded_query(back_translated_query[0]['translation_text'], [score])
        # return super().get_expanded_query(q, [0])

    # Returns the name of the model ('backtranslation) with name of the target language
    # Example: 'backtranslation_fra_latn'
    def get_model_name(self):
        return super().get_model_name() + '_' + self.tgt.lower()

    # Calculate the difference between the original and back-translated query
    def semsim(self, q1, q2):
        me, you = self.transformer_model.encode([q1, q2])
        return 1 - cosine(me, you)


if __name__ == "__main__":
    qe = BackTranslation()
    print(qe.get_model_name())
    print(qe.get_expanded_query('This is my pc'))
