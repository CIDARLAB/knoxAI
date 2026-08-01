from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, SpearmanCorrCoef, KendallRankCorrCoef, PearsonCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

class SharedStepsMixin:
    def setup_metrics(self):
        # Regression
        self.r2score  = R2Score()
        self.rmse     = MeanSquaredError(squared=False)
        self.mae      = MeanAbsoluteError()
        #self.spearman = SpearmanCorrCoef()
        self.kendall  = KendallRankCorrCoef()
        self.pearson  = PearsonCorrCoef()

        # Classification
        self.acc      = BinaryAccuracy()
        self.prec     = BinaryPrecision()
        self.rec      = BinaryRecall()
        self.f1       = BinaryF1Score()
        self.auroc    = BinaryAUROC()

    # -------------------------
    # TRAINING
    # -------------------------
    def training_step(self, batch):
        y_hat = self(batch).view(-1)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # -------------------------
    # VALIDATION
    # -------------------------
    def validation_step(self, batch):
        y_hat = self(batch).view(-1)
        val_loss = self.loss_fn(y_hat, batch.y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.compute_metrics(y_hat, batch.y, stage="val")

    # -------------------------
    # TESTING
    # -------------------------
    def test_step(self, batch):
        y_hat = self(batch).view(-1)
        test_loss = self.loss_fn(y_hat, batch.y)
        self.log(f"test_{self.loss_name}", test_loss, on_epoch=True)
        self.compute_metrics(y_hat, batch.y, stage="test")

    # -------------------------
    # METRICS
    # -------------------------
    def compute_metrics(self, y_hat, y, stage):

        if self.task == "regression":
            self.r2score.update(y_hat, y)
            self.rmse.update(y_hat, y)
            self.mae.update(y_hat, y)
            #self.spearman.update(y_hat, y)
            self.kendall.update(y_hat, y)
            self.pearson.update(y_hat, y)

            self.log(f"{stage}_R2",       self.r2score, on_epoch=True)
            self.log(f"{stage}_RMSE",     self.rmse,    on_epoch=True)
            self.log(f"{stage}_MAE",      self.mae,     on_epoch=True)
            #self.log(f"{stage}_Spearman", self.spearman, on_epoch=True)
            self.log(f"{stage}_Kendall",  self.kendall,  on_epoch=True)
            self.log(f"{stage}_Pearson",  self.pearson,  on_epoch=True)

        elif self.task == "classification":
            self.log(f"{stage}_Accuracy",  self.acc(y_hat, y),   on_epoch=True)
            self.log(f"{stage}_Precision", self.prec(y_hat, y),  on_epoch=True)
            self.log(f"{stage}_Recall",    self.rec(y_hat, y),   on_epoch=True)
            self.log(f"{stage}_F1",        self.f1(y_hat, y),    on_epoch=True)
            self.log(f"{stage}_AUROC",     self.auroc(y_hat, y), on_epoch=True)

        elif self.task == "ranking":
            self.log(f"{stage}_Spearman", self.spearman(y_hat, y), on_epoch=True)
            self.log(f"{stage}_Kendall",  self.kendall(y_hat, y),  on_epoch=True)
            self.log(f"{stage}_Pearson",  self.pearson(y_hat, y),  on_epoch=True)
