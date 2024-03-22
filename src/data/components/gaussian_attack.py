from secml.data import CDataset
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks.evasion import CAttackEvasion

from secml.array import CArray
from torch import randn

class GaussianAttack(CAttackEvasion):
    """
    secml subclass for a Gaussian attack. It is a simple attack that adds Gaussian noise to the input data.

    Modifications from original CAttackEvasion class:
    - Avoid _run() method bc of its simplicity.
    - No class selection, all classes are attacked.
    - adv_f_obj is a mean of the f_obj. I set to None, as it is not used for the attack.

    Args:
        classifier: Target classifier.
        epsilons: 3*standard deviation of the Gaussian noise. The values of epsilon can be compared with the extent of the perturbations of a PGD attack.

    """
    def __init__(self, 
                classifier: CClassifierPyTorch, 
                epsilons:float = 1.0):
        super(CAttackEvasion, self).__init__(classifier)

        self.attacked_classifier = classifier
        self.epsilon = epsilons

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
        noise = CArray(randn(X.shape[0], X.shape[1])*self.epsilon/3).clip(-self.epsilon,self.epsilon)
        adv_X = X + noise.abs()

        Y = CArray(Y).atleast_2d()
        adv_ds = CDataset(adv_X.deepcopy(), Y.deepcopy())

        adv_Y_pred, adv_scores = self.attacked_classifier.predict(adv_ds.X, return_decision_function=True)
        adv_Y_pred = CArray(adv_Y_pred)

        adv_f_obj = None # try
        return adv_Y_pred, adv_scores, adv_ds, adv_f_obj


if __name__ == "__main__":
    GaussianAttack()