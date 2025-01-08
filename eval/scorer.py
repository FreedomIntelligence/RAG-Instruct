from collections import defaultdict
import re
import json
import difflib

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity >= highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index

def match_choice3(text,options):
    matches = list(re.finditer(r"(is |是|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )([abcdefghijklmnABCDEFGHIJKLMN])(\W|$)", text, re.S))
    if matches:
        ans = matches[0].group(2)
        return ans,1

    text = text.lower()

    opsindex = [(opt,text.rindex(options[opt].lower())) for opt in options if options[opt].lower() in text]
    if len(opsindex) > 0:
        return sorted(opsindex,key=lambda x:x[1],reverse=True)[0][0],2
    
    oplabels = [x for x in options]
    opans = [options[x].lower() for x in options]
    ansindex = find_most_similar_index(opans,text.lower())
    return oplabels[ansindex], 3

def match(prediction, ground_truth):
    for gt in ground_truth:
        matchres = re.search(r"(\W|^)("+re.escape(gt)+r")(\W|$)",prediction.lower(),re.S)
        if matchres:
            return 1
    return 0

def score(data):
    res = {}
    wrong_data = []
    cor_data = []
    for da in data:
        if da['source'] not in res:
            res[da['source']] = [0,0,0]

        if da['source'] in ['openbookqa_processed_retrieved','arc_challenge_processed','medqa_test_en_retrieved']:
            if 'options' not in da:
                if 'choices' in da:
                    da['options'] = {la:te for te, la in zip(da['choices']['text'],da['choices']['label'])}
            
            if '## Response:' in da['output']:
                output = da['output'].split('## Response:')[1]
            else:
                output = da['output']
            
            ans,ans_type = match_choice3(output,da['options'])
            da['ans'] = ans
            da['ans_type'] = ans_type

            if ans in da['golds']:
                res[da['source']][1] += 1
                cor_data.append(da)
            else:
                wrong_data.append(da)
            res[da['source']][2] += 1

        elif da['source'] in ['2WIKI_test_processed','triviaqa_test_w_gs','webqa_processed_retrieved','musique_ans_dev','popqa_longtail_w_gs', "triviaqa_helpful", "triviaqa_mid_help", "triviaqa_helpless"]:
            gold = [x.lower() for x in da['golds']]
            if match(da['output'].lower(),gold):
                res[da['source']][1] += 1
                cor_data.append(da)
            else:
                wrong_data.append(da)
            res[da['source']][2] += 1

        elif da['source'] in ['ConvFinQA_test_retrieved']:
            gold = [x.lower() for x in da['golds']]
            match_test  = da['output'].lower()
            
            effect_len = 4
            testgold = gold[0].replace('0','').replace('.','').replace(',','')
            if len(testgold) > 1:
                gold = [testgold]
                match_test = match_test.replace('0','').replace('.','').replace(',','')
                
            if match(match_test,gold):
                res[da['source']][1] += 1
                cor_data.append(da)
            else:
                wrong_data.append(da)
            res[da['source']][2] += 1

        elif da['source'] in ['pubmedqa_test_retrieved','health_claims_processed','hotpot_dev_ori', "hotpot_helpful", "hotpot_helpful", "hotpot_mid_help", "hotpot_helpless"]:
            gold = [x.lower() for x in da['golds']]
            if len(da['golds']) == 1 and da['golds'][0] in ['yes','no','false','true','maybe']:
                onegold = da['golds'][0]
                matchres = re.search(r"(\W|^)("+re.escape(onegold)+r")(\W|$)",da['output'].lower(),re.S)
                if matchres:
                    res[da['source']][1] += 1
                    cor_data.append(da)
                else:
                    wrong_data.append(da)
                res[da['source']][2] += 1
            else:
                if match(da['output'].lower(),gold):
                    res[da['source']][1] += 1
                    cor_data.append(da)
                else:
                    wrong_data.append(da)
                res[da['source']][2] += 1
        
        else:
            raise ValueError('wrong')
        
    for k in res:
        res[k][0] = res[k][1] / res[k][2]
    
    source_num = defaultdict(int)
    each_num = 20
    save_data = []
    for da in wrong_data:
        if source_num[da['source']] == each_num:
            continue
        source_num[da['source']] += 1
        if 'input_str' not in da:
            da['input_str'] = ''
        tmp = {'input_str':da['input_str'],'instruction':da['instruction'],'output':da['output'],'golds':da['golds'],'source':da['source']}
        save_data.append(tmp) 
    wrong_data = save_data

    source_num = defaultdict(int)
    each_num = 20
    save_data = []
    for da in cor_data:
        if source_num[da['source']] == each_num:
            continue
        source_num[da['source']] += 1
        if 'input_str' not in da:
            da['input_str'] = ''
        tmp = {'input_str':da['input_str'],'instruction':da['instruction'],'output':da['output'],'golds':da['golds'],'source':da['source']}
        save_data.append(tmp) 
    cor_data = save_data
    
    return res,wrong_data,cor_data
