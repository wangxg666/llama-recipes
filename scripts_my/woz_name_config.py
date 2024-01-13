name2name = {
    "rosa's bed and breakfast":"rosas bed and breakfast",
    "the cambridge belfry":"cambridge belfry",
    "the fitzwilliam museum":"fitzwilliam museum",
    "sheep's green and lammas land park fen causeway":"sheeps green and lammas land park fen causeway",
    "the cambridge chop house":"cambridge chop house",
    "the nirala":"nirala",
    "kettle's yard":"kettles yard",
    "king's college":"kings college",
    "christ's college":"christ college",
    "the lensfield hotel":"lensfield hotel",
    "museum of archaelogy and anthropology":"museum of archaeology and anthropology"
}



import json
service2slot_keys = json.load(open('woz_valid_slot_keys.json'))


def update_slots(service2slot_kvs):
    output_service2slot_kvs = {}
    for service, slot_kvs in service2slot_kvs.items():
        if service not in service2slot_keys:
            continue
        output_slot_kvs = {}
        for slot_key, slot_val in slot_kvs.items():
            if slot_key not in service2slot_keys[service]:
                continue
            if slot_val in name2name:
                slot_val = name2name[slot_val]
            output_slot_kvs[slot_key] = slot_val
        if output_slot_kvs:
            output_service2slot_kvs[service] = output_slot_kvs
    return output_service2slot_kvs
