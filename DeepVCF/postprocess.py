from .biopandas import pandas as pd
from .tools import pathing, Path


class Metrics:

    def __init__(self, real_vcf, predicted_vcf):
        self.real_vcf = pd.read_vcf(real_vcf) if isinstance(real_vcf, (str, Path)) else real_vcf
        self.predicted_vcf = pd.read_vcf(predicted_vcf) if isinstance(predicted_vcf, (str, Path)) else predicted_vcf
        complete_real, real_hom_alt, real_het = self.pull_simple_snps(self.real_vcf)
        complete_pred, pred_hom_alt, pred_het = self.pull_simple_snps(self.predicted_vcf)
        tn, fp, fn, tp = self.confusion_matrix(set(real_hom_alt.POS), set(pred_hom_alt.POS), set(complete_real.POS), set(complete_pred.POS)) 
        _tn, _fp, _fn, _tp = self.confusion_matrix(set(real_het.POS), set(pred_het.POS), set(complete_real.POS), set(complete_pred.POS))
        self.metrics = {
            'hom_alt': self.get_metrics(len(tn), len(fp), len(fn), len(tp)),
            'het':  self.get_metrics(len(_tn), len(_fp), len(_fn), len(_tp)),
        }

    def get_metrics(self, tn, fp, fn, tp):
        metrics = {
            'Sensitivity': self.sensitivity(tp, fn),
            'PPV': self.precision(tp, fp),
            'Accuracy': self.accuracy(tp, tn, fp, fn),
            'F1': self.f1(tp, fp, fn),
        }
        return metrics

    @staticmethod
    def pull_simple_snps(df):
        df = df.copy()
        df['GT'] = df.iloc[:, -1].apply(lambda d: d.get('GT', '').replace('|', '/'))
        no_indels =  df[~(df.REF.str.len() > 1)&~(df.ALT.str.len() > 1)]
        hom_alt = no_indels[no_indels.GT=='1/1']
        het = no_indels[(no_indels.GT=='0/1')|(no_indels.GT=='1/0')]
        return df, hom_alt, het

    @staticmethod
    def confusion_matrix(real, pred, complete_real, complete_pred):
        tp = real & pred
        fp = pred - real
        tn = (complete_real - real) & (complete_pred - pred)
        fn = real - pred 
        return tn, fp, fn, tp
    
    @staticmethod
    def sensitivity(tp, fn):
        return tp / (tp + fn)
    
    @staticmethod
    def precision(tp, fp):
        return tp / (tp + fp)

    @staticmethod
    def f1(tp, fp, fn):
        return (2*tp) / ( (2*tp) + fn + fp)

    @staticmethod
    def accuracy(tp, tn, fp, fn):
        return (tp+tn) / (tp+tn+fp+fn)