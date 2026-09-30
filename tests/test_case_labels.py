from cus_ai.case_labels import import_case_inventory

def test_import_keeps_owner_label_and_leaves_test_unlabeled():
    files = [{'id': 'a', 'path': 'Common preterm brain injury and abnormalities/Grade 1/case/coronal.nii.gz', 'size_bytes': 42},
             {'id': 'b', 'path': 'Preterm brain injury test cases/Case 1/sagittal.nii.gz', 'size_bytes': 42}]
    result = import_case_inventory(files)
    assert result['category_labeled_files'] == 1
    assert result['unlabeled_test_files'] == 1
    assert result['records'][0]['source_category_label'] == 'Grade 1'
    assert result['records'][0]['paper_domain_labels'] == {}
    assert result['records'][1]['infant_code'] is None
    assert len(result['same_size_candidates']) == 1

def test_legacy_grade_four_does_not_become_current_gmh_grade():
    result = import_case_inventory([{'id': 'a', 'path': 'Common preterm brain injury and abnormalities/Grade 4/source.nii.gz', 'size_bytes': 42}])
    assert 'PVHI' in result['records'][0]['mapping_note']
    assert result['records'][0]['paper_domain_labels'] == {}
