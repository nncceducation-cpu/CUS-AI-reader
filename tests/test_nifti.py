import nibabel as nib
import numpy as np
import pytest
from cus_ai.media import DecodeLimits, decode_media, decode_media_path

def test_nifti_preserves_every_frame_and_shared_intensity_scale(tmp_path):
    data = np.stack([np.full((8, 7), v, dtype=np.int16) for v in (10, 40, 90)], axis=2)
    path = tmp_path / 'sweep.nii.gz'
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    result = decode_media(path.name, path.read_bytes())
    assert [f.frame_index for f in result.frames] == [0, 1, 2]
    assert [np.asarray(f.image)[0, 0] for f in result.frames] == [10, 40, 90]
    assert result.technical_metadata['all_frames_processed']
    assert all(f.pixel_spacing_mm is None for f in result.frames)
    with pytest.raises(ValueError, match='budget'):
        decode_media_path(path, DecodeLimits(max_frames=2))

def test_nifti_rejects_ambiguous_four_dimensions(tmp_path):
    path = tmp_path / 'volume.nii'
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5, 2), dtype=np.uint8), np.eye(4)), path)
    with pytest.raises(ValueError, match='additional axes'):
        decode_media_path(path)

def test_unverified_nifti_abstains_even_with_validated_feature_manifest():
    from PIL import Image
    from cus_ai.media import MediaFrame
    from cus_ai.model import ModelManifest, build_study_prediction
    manifest = ModelManifest('test', '1', 'test.onnx', (32, 32),
                             ['plane_coronal', 'plane_sagittal'], validated=True)
    frames = [MediaFrame('volume.nii', i, Image.new('L', (32, 32)), 'nifti') for i in range(2)]
    result = build_study_prediction(manifest, frames, np.array([[1., 0.], [0., 1.]]))
    assert result.abstained
    assert any('NIfTI' in reason for reason in result.abstention_reasons)
