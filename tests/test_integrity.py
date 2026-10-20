import csv
import pytest

from histchess.core import CAST_POS,CAST_NEG,pair_from_moves
from histchess.audit import audit_final_predictions
from histchess.experiments import write_csv


def test_final_audit_rejects_duplicate_bad_probability_and_mismatched_labels(tmp_path):
    p=pair_from_moves('castling',CAST_POS,CAST_NEG)
    p['split']='test'
    base=[dict(seed=1,history_moves='0',pair_id=p['pair_id'],
               variant=v,task='castling',label=int(v=='positive'),
               score=float(v=='positive'),prediction=int(v=='positive'))
          for v in ('positive','negative')]
    dest=tmp_path/'gru_predictions.csv'
    for variant in [base+base[:1],
                    [dict(base[0],score='nan'),base[1]],
                    [dict(base[0],label=0),base[1]],
                    [dict(base[0],prediction=0),base[1]]]:
        write_csv(dest,variant)
        with pytest.raises(ValueError):
            audit_final_predictions([p],tmp_path,(1,),(0,),skip_llm=True)
