"""CPU contract tests. Pattern fixtures are NOT cinematic artwork or visual approval evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import wave

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from bootstrap import materialize, ROOT, sources
ENGINE = tempfile.TemporaryDirectory()
sys.path.insert(0, str(materialize(ENGINE.name, studio=True)))
import storybook as SB, storybook_atlas as SA, cinematic_workflow as CW, studio_run, studio
import storybook_scene as SS
from preview import render_review
from deliver import build_video


def fixture(root, cid, version='test-v2', approved=True, story=False):
    src = root / ('source-' + cid + version); src.mkdir()
    # A colorful texture makes it possible to detect source-image survival and pose changes.
    arr = np.zeros((180,100,4),dtype=np.uint8)
    arr[:,:,:3] = np.indices((180,100))[1][:,:,None] * np.array([2,1,1],dtype=np.uint8)
    arr[:,:,3] = 255
    Image.fromarray(arr).save(src/'idle.png')
    arr[:50,:,1]=230;Image.fromarray(arr).save(src/'point.png')
    Image.new('RGBA',(40,30),(230,20,160,255)).save(src/'mouth.png')
    Image.new('RGBA',(40,20),(20,230,120,255)).save(src/'blink.png')
    spec={'poses':{p:{'file':p+'.png','feet':[50,179],'top':0,'head':[50,0,60],
                     'view':'front','face':{'box':[30,10,70,50]}} for p in ['idle','point']},
          'pose_map':{p:'point' if p=='point' else 'idle' for p in SB.POSES},
          'face':{'anchor_pose':'idle','box':[30,10,70,50],
                  'blink':{'file':'blink.png','box':[30,10,70,30]},
                  'visemes':{v:{'file':'mouth.png','box':[30,30,70,60]} for v in SB.VISEMES if v!='rest'},
                  'expressions':{'surprised':{'file':'blink.png','box':[30,10,70,30]}}}}
    if story:
        # Synthetic coverage for the production story's glasses, popcorn and rear views.
        spec['face']['wear'] = {'3d_glasses': {'file':'blink.png','box':[30,10,70,30]}}
        for view in ('front', 'back'):
            for hold in (None, 'popcorn', '3d_glasses'):
                pose = copy.deepcopy(spec['poses']['idle'])
                pose.update(view=view, hold=hold, base='hold' if hold else 'idle')
                spec['poses'][f'{view}-{hold}'] = pose
    d=SA.import_atlas(src,root,cid,version,spec,{'model':studio.FLUX_MODEL,'test_fixture':True},log=lambda *a:None)
    (d/'previews').mkdir();(d/'previews/fixture.txt').write_text('test only')
    if approved: SB.approve(d,'automated test fixture, not human production approval')
    return d


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        SB._LIB.clear();SS._ATLAS_P.clear()
    def tearDown(self): self.tmp.cleanup()
    def pair(self,approved=True):
        for cid in ['arjun','meera']:fixture(self.root,cid,approved=approved)
    def test_cinematic_requires_approved_atlas(self):
        self.pair(False)
        with self.assertRaises(ValueError): CW.bind_plan({},self.root,{'arjun':'test-v2','meera':'test-v2'})
    def test_binding_preserves_story_and_audio(self):
        self.pair();old={'shots':[{'lines':[{'text':'keep','voice':'original'}]}],'environments':{'cinema':'v1'}}
        new=CW.bind_plan(old,self.root,{'arjun':'test-v2','meera':'test-v2'})
        self.assertEqual(old['shots'],new['shots']);self.assertNotIn('cinematic_characters',old)
        self.assertTrue(new['cinematic_characters'])
    def test_cached_load_rechecks_pins(self):
        fixture(self.root,'arjun');SB.load('arjun','test-v2',self.root)
        with self.assertRaises(ValueError):SB.load('arjun','test-v2',self.root,expect_sha='wrong')
        with self.assertRaises(ValueError):SB.load('arjun','test-v2',self.root,expect_content='wrong')
    def test_cached_asset_tamper_rejected(self):
        d=fixture(self.root,'arjun');SB.load('arjun','test-v2',self.root)
        (d/'idle.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):SB.load('arjun','test-v2',self.root)
    def test_identity_mismatch_rejected(self):
        d=fixture(self.root,'arjun');m=json.loads((d/'manifest.json').read_text());m['character']='other'
        (d/'manifest.json').write_text(json.dumps(m))
        with self.assertRaises(ValueError):SB.load('arjun','test-v2',self.root)
    def test_missing_cinematic_assets_never_draw(self):
        plan={'cinematic_characters':True,'visual_style_version':'storybook_v1','character_assets':{
            c:{'version':'missing','status':'approved','manifest_sha256':'a','content_sha256':'b'} for c in ['arjun','meera']}}
        with patch.object(SB,'build_character',side_effect=AssertionError('procedural fallback')):
            with self.assertRaises(FileNotFoundError):SB.prepare_assets(plan,root=self.root)
    def test_legacy_plan_is_untouched(self): CW.validate_plan({'shots':[]},self.root)
    def test_versions_are_immutable(self):
        fixture(self.root,'arjun')
        with self.assertRaises(FileExistsError):SA.import_atlas(self.root,self.root,'arjun','test-v2',{}, {})
    def test_conversation_view_keeps_face_animation(self):
        ch=MagicMock();ch.atlas_pose.return_value='three_quarter'
        self.assertEqual(SS.atlas_view(ch,'idle','front',.35,(),None),'front')
        self.assertEqual(SS.atlas_view(ch,'idle','back',.35,(),None),'back')
    def test_prototype_cannot_be_approved(self):
        d=fixture(self.root,'arjun',approved=False)
        m=json.loads((d/'manifest.json').read_text());m['provenance']['prototype_only']=True
        (d/'manifest.json').write_text(json.dumps(m))
        with self.assertRaises(ValueError):SB.approve(d,'reviewer')
    def test_full_atlas_requires_animation_approval(self):
        d=self.root/'arjun/atlas';d.mkdir(parents=True);master=d/'idle_front.png';master.write_bytes(b'fixture')
        CW.record_review(self.root/'arjun/design_approval.json',[master],'design','reviewer',True)
        with self.assertRaises(ValueError):studio_run.run_atlas(MagicMock(),self.root,'arjun')
    def test_pose_and_face_art_used(self):
        fixture(self.root,'arjun');ch=SB.load('arjun','test-v2',self.root);p=SA.AtlasPainter(ch,.3)
        a=np.asarray(p._pose_image('idle',{}));b=np.asarray(p._pose_image('point',{}))
        self.assertFalse(np.array_equal(a,b))
        blink=np.asarray(p._pose_image('idle',{'blink':True}))
        mouth=np.asarray(p._pose_image('idle',{'speaking':True,'viseme':'wide'}))
        self.assertFalse(np.array_equal(a,blink));self.assertFalse(np.array_equal(a,mouth))
        self.assertTrue(np.array_equal(a,np.asarray(p._pose_image('idle',{'speaking':False,'viseme':'wide'}))))
        self.assertIsNone(ch.atlas_pose('point',view='back'))
    def test_content_hash_survives_approval(self):
        d=fixture(self.root,'arjun',approved=False);m=json.loads((d/'manifest.json').read_text());h=SB.content_fingerprint(m)
        before=SB.manifest_sha(d/'manifest.json');after=SB.approve(d,'test reviewer')
        self.assertNotEqual(before,after);self.assertEqual(h,SB.content_fingerprint(json.loads((d/'manifest.json').read_text())))
    def test_human_review_binds_files(self):
        f=self.root/'master.png';f.write_bytes(b'original');r=self.root/'approval.json'
        with self.assertRaises(ValueError):CW.record_review(r,[f],'design','test',False)
        CW.record_review(r,[f],'design','test',True);CW.require_review(r,[f],'design')
        f.write_bytes(b'new')
        with self.assertRaises(ValueError):CW.require_review(r,[f],'design')
    def test_reference_and_resume_contract(self):
        ref=self.root/'ref.png';Image.new('RGB',(32,32),'red').save(ref)
        class Gen:
            revision='test';name='mock';calls=0
            def __call__(self,prompt,refs,size,seed):
                self.calls+=1;return Image.open(ref),seed,{}
        gen=Gen();out=self.root/'candidate.png'
        self.assertEqual(studio_run.run_job(gen,out,'a',[ref],(32,32),1,{}),'made')
        self.assertEqual(studio_run.run_job(gen,out,'a',[ref],(32,32),1,{}),'skipped');self.assertEqual(gen.calls,1)
        with self.assertRaises(FileExistsError):studio_run.run_job(gen,out,'changed',[ref],(32,32),1,{})
        with self.assertRaises(FileNotFoundError):studio_run.run_poc(gen,self.root/'poc')
    def test_image_dimensions_forwarded(self):
        fake_torch=MagicMock();pipe=MagicMock()
        pipe.return_value.images=[Image.fromarray(np.random.default_rng(1).integers(0,255,(64,32,3),dtype=np.uint8))]
        with patch.dict(sys.modules,{'torch':fake_torch}):studio.generate(pipe,'test',1,size=640,height=832)
        self.assertEqual(pipe.call_args.kwargs['height'],832);self.assertEqual(pipe.call_args.kwargs['width'],640)
    def test_rhubarb_mapping_and_silence(self):
        self.assertEqual(CW.rhubarb_visemes([{'start':.1,'end':.3,'value':'D'}],10,5),['rest','wide','wide','rest','rest'])
        with self.assertRaises(ValueError):CW.rhubarb_visemes([{'start':0,'end':1,'value':'?'}],10,10)
    def test_generated_notebook_syntax(self):
        from build_notebook import build
        for i,c in enumerate(build()['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'cell {i}','exec')
    def test_video_builder_changes_only_characters(self):
        self.pair();template=ROOT/'colab/notebooks/colab_three_d_friends_te_v3_storybook.ipynb'
        plan=json.loads((ROOT/'colab/plans/three_d_friends_te_v3_storybook_plan.json').read_text())
        out=self.root/'new.ipynb';bound=build_video(template,plan,self.root,{c:'test-v2' for c in ['arjun','meera']},out)
        self.assertEqual(bound['shots'],plan['shots'])
        before=json.loads(template.read_text());after=json.loads(out.read_text())
        changed=[i for i,(a,b) in enumerate(zip(before['cells'],after['cells'])) if a['source']!=b['source']]
        self.assertEqual([after['cells'][i]['metadata']['tags'][0] for i in changed], ['settings', 'plan'])
        # The target already embeds the canonical character modules; only selected versions and plan binding change.
        for i in changed:compile(''.join(after['cells'][i]['source']),f'video cell {i}','exec')
    def test_actual_encoded_movie_and_reuse(self):
        self.pair();audio=[]
        for cid in ['arjun','meera']:
            p=self.root/(cid+'.wav');a=(np.sin(np.arange(8000)*.2)*10000).astype('<i2')
            with wave.open(str(p),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(a.tobytes())
            audio.append(p)
        with patch.dict(sys.modules,{'diffusers':None,'studio_run':None,'studio':None}):
            movie=render_review(self.root,'test-v2',audio,['fixture Arjun','fixture Meera'],self.root/'movie',True,fps=8,size=(320,180))
            second=render_review(self.root,'test-v2',audio,['second scene','reuse'],self.root/'second',True,fps=8,size=(320,180),closeup=True)
        for m in (movie,second):
            data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(m)]))
            self.assertEqual({s['codec_type'] for s in data['streams']},{'video','audio'})
            self.assertTrue(list(m.parent.glob('frame-*.png')))
        self.assertNotEqual(movie.read_bytes(),second.read_bytes())


if __name__=='__main__':unittest.main(verbosity=2)
