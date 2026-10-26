import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from adr.mlp.train import (TrainingConfig, build_model, classification_metrics,
                           counterfactual_metrics, fit_linear, train_mlp)


def config(**overrides):
    return TrainingConfig(prepared=Path('unused'), output_root=Path('unused'), **overrides)


def test_exact_architecture_and_defaults():
    cfg = config()
    model = build_model(cfg)
    assert [(m.in_features,m.out_features) for m in model if isinstance(m,nn.Linear)] == [(4623,256),(256,64),(64,1)]
    assert [m.p for m in model if isinstance(m,nn.Dropout)] == [.2,.2]
    assert sum(isinstance(m,nn.ReLU) for m in model)==2
    assert model(torch.zeros(3,4623)).shape == (3,1)
    assert (cfg.seed,cfg.learning_rate,cfg.weight_decay,cfg.batch_size,cfg.max_epochs,cfg.patience,cfg.threshold)==(42,.001,.0001,32,100,10,.5)


def test_metrics_confusion_order_threshold_and_class_labels():
    y=np.array([0,0,1,1])
    result=classification_metrics(y,np.array([.1,.5,.6,.9]))
    assert result['confusion_matrix']==[[1,1],[0,2]]
    assert result['accuracy']==.75 and result['balanced_accuracy']==.75
    assert result['EXECUTE']['precision']==pytest.approx(2/3)
    assert result['EXECUTE']['recall']==1
    assert result['SKIP']['precision']==1 and result['SKIP']['recall']==.5
    assert result['macro_f1']==pytest.approx((.8+2/3)/2)
    assert result['roc_auc']==1 and result['average_precision_execute']==1
    majority=classification_metrics(y,np.ones(4))
    assert majority['SKIP']['f1']==0 and majority['roc_auc']==.5
    with pytest.raises(ValueError,match='probabilities'):
        classification_metrics(y,np.array([0,1,float('nan'),1]))


def pair_metadata():
    return [{'trajectory_id':'t1','counterfactual_pair_id':'p1','state_index':i,
             'example_id':f'e{i}','label':label,'embedding_keys':{'candidate_subquery':'same'}}
            for i,label in [(0,'EXECUTE'),(3,'SKIP')]]


def test_counterfactual_pairs_and_identical_candidate_ties():
    metadata=pair_metadata()
    result=counterfactual_metrics(metadata,np.array([.8,.2]))
    assert result['n_pairs']==1 and result['both_correct_count']==1
    assert result['earlier_execute_accuracy']==1 and result['later_skip_accuracy']==1
    assert result['mean_earlier_execute_probability']==.8
    assert result['mean_later_execute_probability']==.2
    assert result['probability_decreased_count']==1
    tied=counterfactual_metrics(metadata,np.array([.8,.8]))
    assert tied['both_correct_count']==0 and tied['probability_decreased_count']==0
    tiny=counterfactual_metrics(metadata,np.array([.8,.8-1e-12]))
    assert tiny['probability_decreased_count']==0
    malformed=copy.deepcopy(metadata)
    malformed[1]['embedding_keys']['candidate_subquery']='different'
    with pytest.raises(ValueError,match='Invalid'):
        counterfactual_metrics(malformed,np.array([.8,.2]))
    with pytest.raises(ValueError,match='Invalid'):
        counterfactual_metrics(metadata[:1],np.array([.8]))


def test_early_stopping_restoration_and_cpu_determinism():
    # Toy inputs only. No production trajectory is trained in tests.
    cfg=config(input_dimensions=4,hidden_dimensions=(8,4),learning_rate=0,
               max_epochs=20,patience=2,batch_size=4)
    X=np.arange(32,dtype=np.float32).reshape(8,4)/32
    y=np.array([0,1]*4,dtype=np.int64)
    first=train_mlp(X,y,X[:4],y[:4],cfg)
    second=train_mlp(X,y,X[:4],y[:4],cfg)
    model,history,best,train_p,val_p=first
    assert len(history)==3 and best==1
    assert history==second[1] and np.array_equal(val_p,second[-1])
    assert not model.training
    with torch.no_grad():
        restored=nn.BCEWithLogitsLoss()(model(torch.from_numpy(X[:4])).squeeze(-1),torch.from_numpy(y[:4]).float()).item()
    assert restored==history[best-1]['validation_loss']
    assert all(v.device.type=='cpu' for v in model.parameters())


def test_fixed_logistic_no_weighting_and_no_validation_fit():
    cfg=config()
    X=np.array([[-2.,0],[-1.,0],[1.,0],[2.,0]])
    y=np.array([0,0,1,1])
    model,p,notices=fit_linear(X,y,np.array([[0.,0],[100.,0]]),cfg)
    other,q,_=fit_linear(X,y,np.array([[-100.,0]]),cfg)
    assert np.array_equal(model.coef_,other.coef_)
    assert model.C==1 and model.class_weight is None and model.solver=='lbfgs'
    assert p[0]==pytest.approx(.5) and p[1]>.5 and q[0]<.5
    assert notices==[]
