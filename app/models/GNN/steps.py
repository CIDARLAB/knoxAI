from torchmetrics.regression import MeanAbsoluteError, R2Score, SpearmanCorrCoef, KendallRankCorrCoef, PearsonCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC


class SharedStepsMixin:
    def setup_metrics(self):
        self.r2score = R2Score()
        self.acc = BinaryAccuracy()
        self.spearman = SpearmanCorrCoef()
        
        self.MAE = MeanAbsoluteError()
        self.spearman = SpearmanCorrCoef()
        self.kendall = KendallRankCorrCoef()
        self.pearson = PearsonCorrCoef()
        self.prec = BinaryPrecision()
        self.rec = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.auroc = BinaryAUROC()
        
    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        val_loss = self.loss_fn(y_hat, batch.y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.compute_metrics(y_hat, batch.y, is_testing=False)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        test_loss = self.loss_fn(y_hat, batch.y)
        self.log(self.loss_name, test_loss)
        self.compute_metrics(y_hat, batch.y, is_testing=True)

    def compute_metrics(self, y_hat, y, is_testing=False):
        y_hat = y_hat.view(-1)
        y = y.view(-1)
        
        if not is_testing:
            if self.task == 'regr':
                self.r2score.reset()
                self.r2score.update(y_hat, y)
                self.log("val_R2", self.r2score.compute(), prog_bar=True)
        
            if self.task == 'class':
                self.acc.update(y_hat, y)
                self.log("val_Accuracy", self.acc.compute(), prog_bar=True)
        
            if self.task == 'rank':
                self.spearman.update(y_hat, y)
                self.log("val_Spearmans", self.spearman.compute(), prog_bar=True)

        elif is_testing:
            if self.task == 'regr':
                self.r2score.reset()
                self.r2score.update(y_hat, y)
                self.log("test_R2", self.r2score.compute())

                self.MAE.update(y_hat, y)
                self.log("test_MAE", self.MAE.compute())
                
                self.spearman.update(y_hat, y)
                self.log("test_Spearmans", self.spearman.compute())
                
                self.kendall.update(y_hat, y)
                self.log("test_Kendalls", self.kendall.compute())
                
                self.pearson.update(y_hat, y)
                self.log("test_Pearsons", self.pearson.compute())
    
            if self.task == 'class':
                self.acc.update(y_hat, y)
                self.prec.update(y_hat, y)
                self.rec.update(y_hat, y)
                self.f1.update(y_hat, y)
                self.auroc.update(y_hat, y)

                self.log("test_Accuracy", self.acc.compute())
                self.log("test_Precision", self.prec.compute())
                self.log("test_Recall", self.rec.compute())
                self.log("test_F1", self.f1.compute())
                self.log("test_AUROC", self.auroc.compute())
        
            if self.task == 'rank':
                self.spearman.reset()
                self.spearman.update(y_hat, y)
                self.kendall.update(y_hat, y)
                self.pearson.update(y_hat, y)
                self.log("test_Spearmans", self.spearman.compute())
                self.log("test_Kendalls", self.kendall.compute())
                self.log("test_Pearsons", self.pearson.compute())
                