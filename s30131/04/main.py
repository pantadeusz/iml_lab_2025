# porównanie: RandomForest (sklearn) vs proste DNN (PyTorch) na breast_cancer (maks. skrót)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.optim as optim

# dane + podział
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# baseline: RandomForest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
rf_scores = rf.predict_proba(X_te)[:, 1]
rf_pred = (rf_scores >= 0.5).astype(int)
print("RF:",
      "acc=%.3f" % accuracy_score(y_te, rf_pred),
      "roc_auc=%.3f" % roc_auc_score(y_te, rf_scores),
      "ap=%.3f" % average_precision_score(y_te, rf_scores))

# DNN: skalowanie + MLP
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
Xtr = torch.tensor(X_tr_s, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.long)
Xte = torch.tensor(X_te_s, dtype=torch.float32); yte = torch.tensor(y_te, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))
    def forward(self, x): return self.net(x)

model = MLP(Xtr.shape[1]); opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
for _ in range(20):  # krótki trening full-batch
    model.train(); opt.zero_grad(); out = model(Xtr)
    loss = crit(out, ytr); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    logits = model(Xte)
    prob1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()

print("DNN:",
      "acc=%.3f" % accuracy_score(y_te, pred),
      "roc_auc=%.3f" % roc_auc_score(y_te, prob1),
      "ap=%.3f" % average_precision_score(y_te, prob1))