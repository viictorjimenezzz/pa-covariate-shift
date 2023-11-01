from secml.data import CDataset
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks.evasion import CAttackEvasion

from secml.array import CArray
from torch import randn

class GaussianAttack(CAttackEvasion):
    """
    Modifications from original class:
    - Avoid _run() method bc of its simplicity.
    - No class selection, all classes are attacked.
    - adv_f_obj is a mean of the f_obj. I set to None.
    """
    def __init__(self, 
                classifier: CClassifierPyTorch, 
                epsilons:float = 1.0):
        super(CAttackEvasion, self).__init__(classifier)

        self.attacked_classifier = classifier
        self.noise_std = epsilons/3

    def _run(self):
        return
    
    def f_eval(self):
        return
    
    def grad_eval(self):
        return
    
    def objective_function(self):
        return
    
    def objective_function_gradient(self):
        return

    def run(self, X, Y):
        X = CArray(X).atleast_2d()
        noise = CArray(randn(X.shape[0], X.shape[1])*self.noise_std)
        adv_X = X + noise

        Y = CArray(Y).atleast_2d()
        adv_ds = CDataset(adv_X.deepcopy(), Y.deepcopy())

    
        #adv_scores = self.classifier.forward(adv_X).softmax(dim=1)
        #adv_Y_pred = adv_scores.argmax()
        adv_Y_pred, adv_scores = self.attacked_classifier.predict(adv_ds.X, return_decision_function=True)
        adv_Y_pred = CArray(adv_Y_pred)


        adv_f_obj = None # try
    
        return adv_Y_pred, adv_scores, adv_ds, adv_f_obj


if __name__ == "__main__":
    GaussianAttack()