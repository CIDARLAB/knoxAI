from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, SpearmanCorrCoef, KendallRankCorrCoef, PearsonCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

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

        # MultiClass Classification
        self.mc_acc   = Accuracy(num_classes=self.num_classes, average="macro")
        self.mc_prec  = Precision(num_classes=self.num_classes, average="macro") 
        self.mc_rec   = Recall(num_classes=self.num_classes, average="macro")  
        self.mc_f1    = F1Score(num_classes=self.num_classes, average="macro") 
        self.mc_auroc = AUROC(num_classes=self.num_classes, average="macro")

    # -------------------------
    # TRAINING
    # -------------------------
    def training_step(self, batch):
        y_hat = self.get_y_hat(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # -------------------------
    # VALIDATION
    # -------------------------
    def validation_step(self, batch):
        y_hat = self.get_y_hat(batch)
        val_loss = self.loss_fn(y_hat, batch.y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.compute_metrics(y_hat, batch.y, stage="val")

    # -------------------------
    # TESTING
    # -------------------------
    def test_step(self, batch):
        y_hat = self.get_y_hat(batch)
        test_loss = self.loss_fn(y_hat, batch.y)
        self.log(f"TEST_{self.loss_name}", test_loss, on_epoch=True)
        self.compute_metrics(y_hat, batch.y, stage="TEST")

    # -------------------------
    # Get Predictions
    # -------------------------
    def get_y_hat(self, batch):
        if self.task in ["regression", "classification"]:
            return self(batch).view(-1)
        else:
            return self(batch)

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

        elif self.task == "multiclass_classification":
            self.log(f"{stage}_Accuracy",  self.mc_acc(y_hat, y),   on_epoch=True)
            self.log(f"{stage}_Precision", self.mc_prec(y_hat, y),  on_epoch=True)
            self.log(f"{stage}_Recall",    self.mc_rec(y_hat, y),   on_epoch=True)
            self.log(f"{stage}_F1",        self.mc_f1(y_hat, y),    on_epoch=True)
            self.log(f"{stage}_AUROC",     self.mc_auroc(y_hat, y), on_epoch=True)

        elif self.task == "ranking":
            self.log(f"{stage}_Spearman", self.spearman(y_hat, y), on_epoch=True)
            self.log(f"{stage}_Kendall",  self.kendall(y_hat, y),  on_epoch=True)
            self.log(f"{stage}_Pearson",  self.pearson(y_hat, y),  on_epoch=True)
