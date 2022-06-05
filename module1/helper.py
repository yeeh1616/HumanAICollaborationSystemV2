def setValue(policy, columnName, answer):
    if columnName == 'policy_id':
        policy.policy_id = answer

    elif columnName == 'entry_type':
        policy.entry_type = answer

    elif columnName == 'correct_type':
        policy.correct_type = answer

    elif columnName == 'update_type':
        policy.update_type = answer

    elif columnName == 'update_level':
        policy.update_level = answer

    elif columnName == 'description':
        policy.description = answer

    elif columnName == 'date_announced':
        policy.date_announced = answer

    elif columnName == 'date_start':
        policy.date_start = answer

    elif columnName == 'date_end':
        policy.date_end = answer

    elif columnName == 'country':
        policy.country = answer

    elif columnName == 'ISO_A3':
        policy.ISO_A3 = answer

    elif columnName == 'ISO_A2':
        policy.ISO_A2 = answer

    elif columnName == 'init_country_level':
        policy.init_country_level = answer

    elif columnName == 'domestic_policy':
        policy.domestic_policy = answer

    elif columnName == 'province':
        policy.province = answer

    elif columnName == 'ISO_L2':
        policy.ISO_L2 = answer

    elif columnName == 'city':
        policy.city = answer

    elif columnName == 'type':
        policy.type = answer

    elif columnName == 'type_sub_cat':
        policy.type_sub_cat = answer

    elif columnName == 'type_text':
        policy.type_text = answer

    elif columnName == 'institution_status':
        policy.institution_status = answer

    elif columnName == 'target_country':
        policy.target_country = answer

    elif columnName == 'target_geog_level':
        policy.target_geog_level = answer

    elif columnName == 'target_region':
        policy.target_region = answer

    elif columnName == 'target_province':
        policy.target_province = answer

    elif columnName == 'target_city':
        policy.target_city = answer

    elif columnName == 'target_other':
        policy.target_other = answer

    elif columnName == 'target_who_what':
        policy.target_who_what = answer

    elif columnName == 'target_direction':
        policy.target_direction = answer

    elif columnName == 'travel_mechanism':
        policy.travel_mechanism = answer

    elif columnName == 'compliance':
        policy.compliance = answer

    elif columnName == 'enforcer':
        policy.enforcer = answer

    elif columnName == 'dist_index_high_est':
        policy.dist_index_high_est = answer

    elif columnName == 'dist_index_med_est':
        policy.dist_index_med_est = answer

    elif columnName == 'dist_index_low_est':
        policy.dist_index_low_est = answer

    elif columnName == 'dist_index_country_rank':
        policy.dist_index_country_rank = answer

    elif columnName == 'link':
        policy.link = answer

    elif columnName == 'date_updated':
        policy.date_updated = answer

    elif columnName == 'recorded_date':
        policy.recorded_date = answer

    elif columnName == 'original_text':
        policy.original_text = answer

    elif columnName == 'status':
        policy.status = answer

    return policy


def getValue(policy, columnName):
    res = None

    if columnName == 'policy_id':
        res = policy.policy_id

    elif columnName == 'entry_type':
        res = policy.entry_type

    elif columnName == 'correct_type':
        res = policy.correct_type

    elif columnName == 'update_type':
        res = policy.update_type

    elif columnName == 'update_level':
        res = policy.update_level

    elif columnName == 'description':
        res = policy.description

    elif columnName == 'date_announced':
        res = policy.date_announced

    elif columnName == 'date_start':
        res = policy.date_start

    elif columnName == 'date_end':
        res = policy.date_end

    elif columnName == 'country':
        res = policy.country

    elif columnName == 'ISO_A3':
        res = policy.ISO_A3

    elif columnName == 'ISO_A2':
        res = policy.ISO_A2

    elif columnName == 'init_country_level':
        res = policy.init_country_level

    elif columnName == 'domestic_policy':
        res = policy.domestic_policy

    elif columnName == 'province':
        res = policy.province

    elif columnName == 'ISO_L2':
        res = policy.ISO_L2

    elif columnName == 'city':
        res = policy.city

    elif columnName == 'type':
        res = policy.type

    elif columnName == 'type_sub_cat':
        res = policy.type_sub_cat

    elif columnName == 'type_text':
        res = policy.type_text

    elif columnName == 'institution_status':
        res = policy.institution_status

    elif columnName == 'target_country':
        res = policy.target_country

    elif columnName == 'target_geog_level':
        res = policy.target_geog_level

    elif columnName == 'target_region':
        res = policy.target_region

    elif columnName == 'target_province':
        res = policy.target_province

    elif columnName == 'target_city':
        res = policy.target_city

    elif columnName == 'target_other':
        res = policy.target_other

    elif columnName == 'target_who_what':
        res = policy.target_who_what

    elif columnName == 'target_direction':
        res = policy.target_direction

    elif columnName == 'travel_mechanism':
        res = policy.travel_mechanism

    elif columnName == 'compliance':
        res = policy.compliance

    elif columnName == 'enforcer':
        res = policy.enforcer

    elif columnName == 'dist_index_high_est':
        res = policy.dist_index_high_est

    elif columnName == 'dist_index_med_est':
        res = policy.dist_index_med_est

    elif columnName == 'dist_index_low_est':
        res = policy.dist_index_low_est

    elif columnName == 'dist_index_country_rank':
        res = policy.dist_index_country_rank

    elif columnName == 'link':
        res = policy.link

    elif columnName == 'date_updated':
        res = policy.date_updated

    elif columnName == 'recorded_date':
        res = policy.recorded_date

    elif columnName == 'original_text':
        res = policy.original_text

    elif columnName == 'status':
        res = policy.status

    return res