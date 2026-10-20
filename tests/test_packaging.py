"""ZIP integrity verification catches bad/missing/partial experiment artifacts."""
import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from scripts.verify_artifacts import verify,REQUIRED


def test_output_verifier_rejects_broken_zip(tmp_path):
    path=tmp_path/'broken.zip';path.write_text('no zip')
    with pytest.raises(zipfile.BadZipFile):verify(path)


def test_output_verifier_rejects_missing_manifest(tmp_path):
    path=tmp_path/'partial.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('x/manifest.json','{}')
    with pytest.raises(ValueError,match='Missing report outputs'):verify(path)


def test_output_verifier_rejects_falsified_completion(tmp_path):
    path=tmp_path/'falsified.zip'
    pairs=b'{"pair_id":"p"}\n'
    digest=hashlib.sha256(pairs).hexdigest()
    data={p:b'x' for p in REQUIRED}
    data.update({
        'run_status.json':json.dumps({'completed':False,'paper_complete':False}).encode(),
        'manifest.json':json.dumps({'pairs_sha256':digest,'status':'INCOMPLETE','paper_complete':False}).encode(),
        'final_output_audit.json':json.dumps({'passed':True,'test_pairs':1}).encode(),
        'data/pairs.jsonl':pairs,
        'validation.json':json.dumps({'passed':True}).encode()})
    with zipfile.ZipFile(path,'w') as z:
        for p,b in data.items():z.writestr('test/'+p,b)
    with pytest.raises(ValueError):verify(path)


def test_verifier_accepts_matching_fixture_and_rejects_hash_change(tmp_path):
    from scripts.verify_artifacts import LLM_FILES
    path=tmp_path/'complete.zip'
    pairs=b'{"pair_id":"a"}\n{"pair_id":"b"}\n{"pair_id":"c"}\n'
    png=b'\x89PNG\r\n\x1a\nfixture'
    files={p:b'dummy nonempty fixture' for p in REQUIRED|LLM_FILES}
    files.update({
        'run_status.json':json.dumps({'completed':True,'paper_complete':True}).encode(),
        'manifest.json':json.dumps({'pairs_sha256':hashlib.sha256(pairs).hexdigest(),
                                    'status':'COMPLETE','paper_complete':True}).encode(),
        'final_output_audit.json':json.dumps({'passed':True,'test_pairs':1,
                                              'gru_predictions_checked':12}).encode(),
        'data/pairs.jsonl':pairs,
        'validation.json':json.dumps({'passed':True}).encode(),
        'figures/history_length_ablation.png':png,
        'figures/validation_loss.png':png})
    with zipfile.ZipFile(path,'w') as z:
        for name,blob in files.items():z.writestr('histchess_results/'+name,blob)
    assert verify(path)['passed']
    assert verify(path)['gru_predictions']==12
    files['manifest.json']=json.dumps({'pairs_sha256':'bad',
           'status':'COMPLETE','paper_complete':True}).encode()
    with zipfile.ZipFile(path,'w') as z:
        for name,blob in files.items():z.writestr('histchess_results/'+name,blob)
    with pytest.raises(ValueError,match='SHA256 mismatch'):verify(path)
