"""Preserve owner-supplied folder labels without inventing clinical details."""
from collections import defaultdict

UNLABELED_COLLECTION = "Preterm brain injury test cases"

def import_case_inventory(files: list[dict]) -> dict:
    records = []
    sizes = defaultdict(list)
    for source in files:
        parts = source['path'].split('/')
        if len(parts) < 3:
            raise ValueError('Expected collection/category/source path')
        labeled = parts[0] != UNLABELED_COLLECTION
        record = {
            'source_id': source['id'], 'source_path': source['path'],
            'source_url': source.get('url'), 'size_bytes': source['size_bytes'],
            'collection': parts[0], 'source_category_label': parts[1] if labeled else None,
            'label_status': 'owner_confirmed_category' if labeled else 'unlabeled_test_case',
            'examination_folder': '/'.join(parts[:-1]),
            'infant_code': None, 'study_code': None, 'center': None, 'split': None,
            'source_sha256': None, 'modality': None,
            'paper_domain_labels': {},
            'domain_mapping_status': 'requires_mapping_without_assuming_laterality_or_negative_domains',
        }
        if parts[0].startswith('Common preterm') and parts[1] == 'Grade 4':
            record['mapping_note'] = 'Confirm PVHI and associated GMH-IVH separately; legacy Grade 4 is not a current GMH-IVH grade.'
        records.append(record)
        sizes[source['size_bytes']].append(source['path'])
    return {
        'files': len(records),
        'category_labeled_files': sum(r['source_category_label'] is not None for r in records),
        'unlabeled_test_files': sum(r['source_category_label'] is None for r in records),
        'note': 'Owner confirms category labels. Examination folders are not verified infant identifiers. Equal size is only a duplicate candidate; hash source bytes.',
        'records': records,
        'same_size_candidates': [paths for paths in sizes.values() if len(paths) > 1],
    }
