# region bounding box overlap
#from stanfordcorenlp import StanfordCoreNLP
import json
from collections import namedtuple
from nltk.corpus import propbank
from collections import OrderedDict
#from autocorrect import spell

from nltk.corpus import wordnet as wn

noun_tags = ['NN','NNS']
verb_tags = ['VB','VBD', 'VBG','VBN','VBZ', 'VBP']
attributive_verbs = ['has', 'have']
preposition_tag = ['IN','RP']


#nlp = StanfordCoreNLP(r'/Users/thilinicooray/sem_img/stanford-corenlp-full-2017-06-09/')
#nlp = StanfordCoreNLP('http://localhost', port=9000)

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def intersect(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0.0

def get_verlap_ratio(region1,region2,obj):
    obj_region_intsct = intersect(region1, region2)
    print('overlapping area : ', obj_region_intsct)

    if obj_region_intsct == 0.0:
        return 0.0

    object_overlap_ratio = obj_region_intsct / float(obj['w']*obj['h'])
    print('overlap ratio:', object_overlap_ratio)
    return object_overlap_ratio

# same object with multiple overlapping regions. remove them.
'''def clean_objects(object_list):
    object_dict = {}
    for obj in object_list:
        if obj['names'][0] not in object_dict.keys():
            object_dict[obj['names'][0]] = obj
        else :
            exicting_object = object_dict[obj['names'][0]]
            exicting_object_box = Rectangle(exicting_object['x'], exicting_object['y'], exicting_object['x'] + exicting_object['w'], exicting_object['y'] + exicting_object['h'])
            obj_box = Rectangle(obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h'])
            overlap_with_new = get_verlap_ratio(obj_box, exicting_object_box,obj)
            overlap_with_exsiting = get_verlap_ratio(obj_box, exicting_object_box,exicting_object)
            #select the object which has highest overlap as the common object
            if overlap_with_new <= overlap_with_exsiting:
                object_dict[obj['names'][0]] = obj

    return object_dict.values(),object_dict.keys()'''

def get_complete_relation (key, relation):
    complete_relation = OrderedDict()
    complete_relation['predicate'] =  key
    for key1, value in relation[0].iteritems():
        complete_relation[key1] = value[1]

    return complete_relation

def get_final_semantic_map (semantic_map):
    final_relation_list = []
    for key in semantic_map.keys():
        tuple_list = semantic_map[key]
        sorted_by_length_list = [k for k in sorted(tuple_list, key=lambda k: len(k[0].keys()), reverse=True)]

        unique_relation_list = []

        for relation in sorted_by_length_list:
            complete_relation = get_complete_relation(key, relation)
            if len(unique_relation_list) == 0:
                unique_relation_list.append({'region_id' : relation[1], 'region_relations' : complete_relation})
            else:
                is_subset_found = False
                for added_relation in unique_relation_list:
                    setA = set(complete_relation.items())
                    setB = set(added_relation['region_relations'].items())
                    if setA.issubset(setB):
                       is_subset_found = True
                       break

                if not is_subset_found:
                    unique_relation_list.append({'region_id' : relation[1], 'region_relations' : complete_relation})

        final_relation_list.extend(unique_relation_list)

    return final_relation_list

from gensim.models.keyedvectors import KeyedVectors

data_path = '/home/ta/Projects/SUTD/VSRL/res_data/word2vec/GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(data_path, binary=True)
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import numpy as np

def get_srl_dict(sense,srl_dict,objects):

    # print (srl_dict)
#    non_action_verbs = ['is', 'are']
#    #above verbs used for attributes
#    if srl_dict['predicate'] in non_action_verbs:
#        return
#
#    #get base form of predicate
#    #lmtzr = WordNetLemmatizer()
#    #lemma = lmtzr.lemmatize(srl_dict['predicate'],'v')
#    props={'annotators': 'lemma','pipelineLanguage':'en','outputFormat':'json'}
#    annotation = nlp.annotate(text, properties=props)
#    #print('lemma', annotated)
#    lemma_annotation = json.loads(annotation)
#    text_split = text.split()
#    predicate_index =  [i for i in range(len(text_split)) if text_split[i] == srl_dict['predicate']][0]
#
#    print ('found predicate ', text_split[predicate_index])
#    lemma = lemma_annotation['sentences'][0]['tokens'][predicate_index]['lemma']
    #get semantic role names from propbank
    '''with codecs.open('/Users/thilinicooray/sem_img/propbank-frames/frames/' + lemma +'.xml', 'r', 'latin-1') as fd:
        doc = xmltodict.parse(fd.read())

    verb = lemma
    predicates = doc['frameset']['predicate']
    role_dict = {}
    for predicate in predicates:
        #print(predicate)
        if predicate['@lemma'] == verb:
            #print(predicate['roleset'])
            #ignore several verb senses
            roleset = None
            if isinstance(predicate['roleset'], list):
                roleset = predicate['roleset'][0]
            else:
                roleset = predicate['roleset']
            roles = roleset['roles']
            for role in roles['role']:
                name = role['@descr'].split(',')[0]
                arg = 'ARG' + role['@n']
                role_dict[arg] = name
            break'''

    region_tag_tuples = []
    try:
        roleset = propbank.roleset(sense)
    except ValueError:
        print('no matching frames for predicate ', sense)
        return  None, None
    #only continue if propbank has a frame
    role_dict = {}
    for role in roleset.findall("roles/role"):
        arg = 'A'+ role.attrib['n']
        name = role.attrib['descr'].split(',')[0]
        role_dict[arg] = name

    print ('role_dict=', role_dict)
    print ('role dict from srl tool :' , srl_dict)
    #add place, time and manner
    role_dict[u'AM-LOC'] = 'place'
    role_dict[u'AM-TMP'] = 'time'
    role_dict[u'AM-MNR'] = 'manner'
    role_dict[u'AM-DIR'] = 'direction'
    role_dict[u'AM-ADV'] = 'adverbial modification'
    role_dict[u'AM-DIS'] = 'doscourse marker'
    role_dict[u'AM-EXT'] = 'extent'
    role_dict[u'AM-MOD'] = 'general modification'
    role_dict[u'AM-NEG'] = 'negation'
    role_dict[u'AM-PNC'] = 'proper noun component'
    role_dict[u'AM-PRD'] = 'secondary predicate'
    role_dict[u'AM-PRP'] = 'purpose'
    role_dict[u'AM-REC'] = 'reciprocal'
    mapped_srl_dict = OrderedDict()
    mapped_srl_dict['predicate'] = sense.split('.')[0]
    #keep only NN in value fields of roles

    for key, comp_value in srl_dict.iteritems():
        if key != 'predicate' and key != 'V':
            #to avoid issues where srl tool given roles which do not actually exist in propbank frame roles
            if key not in role_dict:
                return None, None

            key = role_dict[key]
            #without this, can't map objects later
            # print ('key now=', key)
            print('tags needs for ', comp_value, region_tag_tuples)
            tag_tuples, region_tag_tuples = get_tag_tuples(comp_value, region_tag_tuples)
            if tag_tuples is None:
                return None, None
            print(tag_tuples)

            # if value contains a predicate, split value from that place and only consider before that
            # if the remaining string is blank, remove that key from the srl
            verb_removed_tag_tuples = []
            verb_removed_comp = ''
            verb_found = False
            preposition = ''
            for (word,tag) in tag_tuples:
                if tag in verb_tags:
                    verb_found = True
                #pos tagger tag down as adverb, but is it needed for direction
                if tag in preposition_tag or word in ['down', 'up']:
                    preposition = word
                if not verb_found:
                    verb_removed_comp = verb_removed_comp + word + ' '
                    verb_removed_tag_tuples.append((word,tag))

            if len(verb_removed_tag_tuples) == 0:
                continue

            if len(verb_removed_tag_tuples) > 1 :

                value = [word for (word, tag) in verb_removed_tag_tuples
                         if tag in noun_tags]

                if len(value) > 0 :
                    value = value[0]
                # try to map values to actual annotated objects in the image
                # if value ==  annotated entitiry or value part of annotated entity, add the annotated entity
                    object_names = [obj['names'][0] for obj in objects]
                    print ('object_names =', object_names)
                    if value in object_names:
                        mapped_srl_dict[key] = value
                    else:
                        # matching_elements = [obj for obj in object_names if (value in obj or obj in value)]
                        # print('matching elements in the image', matching_elements)
                        # if len(matching_elements)> 0 :
                        #     mapped_srl_dict[key] = matching_elements[0]
                        # else :
                        #     # todo :word similarity needs to be checked if exact match cannot be found
                        #     return  None, None

                        # find similarity based on wordnet nltk
                        # for obj in objects:
                        #     print (obj['synsets'])
                        # object_synsets = [obj['synsets'][0] for obj in objects]
                        # print (object_synsets)

                        value = ' '.join([word for word in value.split() if word not in STOPWORDS])
                        print('Find objects related to', value)
                        # print (value)
                        THRESHOLD_MATCHING = 0.2
                        # matching_elements = map(lambda name: (word2vec.similarity(value, name), name), object_names)
                        matching_elements = []

                        for element in object_names:
                            if len(element.split()) > 1:
                                collect = []
                                for word in element.split():
                                    collect.append(word2vec[word])
                                vector = np.mean( np.array(collect), axis=0 )
                            else:
                                vector = word2vec[element]
                            cosine_similarity = np.dot(vector, word2vec[value]) / (np.linalg.norm(vector) * np.linalg.norm(word2vec[value]))
                            matching_elements.append((cosine_similarity, element))

                        sorted_elements = sorted(matching_elements, key = lambda x: x[0], reverse=True)
                        print ('Sorted_element=', sorted_elements)
                        if sorted_elements[0][0] < THRESHOLD_MATCHING:
                            print ('No object found!')
                            return None, None
                        else:
                            print ('Matching with:', sorted_elements[0][1])
                            mapped_srl_dict[key] = sorted_elements[0][1]

                else:
                    mapped_srl_dict[key] = verb_removed_comp.strip()

            else:
                mapped_srl_dict[key] = verb_removed_comp.strip()

            if preposition:
                mapped_srl_dict[key] = preposition + ',' + mapped_srl_dict[key]

    if len(mapped_srl_dict.keys()) == 1:
        return  None, None

    print('each srl dict ::::' , mapped_srl_dict)
    return mapped_srl_dict, region_tag_tuples

def get_tag_tuples(comp_value, region_tag_tuples):

    k, v = comp_value.items()[0]
    tokens = k.split()
    tags = v.split()

    if len(tokens) != len(tags):
        print('incorrect tagging')
        return None, None

    tuple_list = []

    for i in range(len(tokens)):
        tuple_list.append((tokens[i],tags[i]))
        region_tag_tuples.append((tokens[i],tags[i]))

    print('current tuples', tuple_list, 'region tuples', region_tag_tuples)
    return tuple_list, region_tag_tuples

def srl(srl_map,objects):
    srl_mapping_list = []
#    #execute deep-srl to get semantic role list
#    input_file = 'input.txt'
#    with codecs.open(input_file, 'wt', 'utf-8') as myfile:
#        myfile.write(text)
#    output_file = 'output'+str(region_id) +'.json'
#    #print('deep srl starts....')
#    subprocess.call(shlex.split('sh ./scripts/run_end2end.sh ' + input_file + ' ' + output_file))
#    #print('deep srl ended....')
#    #print("done subprocess")
#    #read output file and load roles with labels
#    if not os.path.exists(output_file):
#        print('SRL output not found')
#        return
#
#    with open(output_file) as srl_file:
#        srl_dict_list = json.load(srl_file)
#
#    for srl_mapping in reversed(srl_dict_list):
#        srl = get_srl_dict(srl_mapping, text, objects)
#        if srl:
#            srl_mapping_list.append(srl)
#
#    os.remove(output_file)
    print ('')
    print ('Going to SRL...')
    print ('srl_map =', srl_map)

    #for each verb
    for element in srl_map:
        print ('verb =', element)
        # print ('what? =', srl_map[element])
        srl, tag_tuples = get_srl_dict(element, srl_map[element], objects)
        print ('srl, tag_tuples =', srl, tag_tuples)
        if srl is not None:
            print('srl', srl, 'tags', tag_tuples)
            srl_mapping_list.append({"srl" : srl, "tags" : tag_tuples})
        print ('Finish this verb: ', element)
        print ('')

    return srl_mapping_list


def get_semantic_map_for_image(regions, objects) :
    semantic_map = {}

    #there are many regions in an image
    for region in regions:
        pos_tagged = []
        parsed_phrase = region['parsed phrase']
        region_id = region['region_id']
        #description = description_raw.lower()
        region_box = Rectangle(region['x'], region['y'], region['x'] + region['width'], region['y'] + region['height'])

        print('parsed region : ', parsed_phrase)
        #spell_corrected = ' '.join(spell(word) for word in description.split())
        # remove membership phrases  (remove has or have from sentence)
        #description = ' '.join(filter(lambda x: x.lower() not in attributive_verbs,  spell_corrected.split()))
        #print ('attribute removed phrases ', description)
        #pos_tagged = nlp.pos_tag(description)
        #print('postagged : ', pos_tagged)

        # consider only descriptions with verbs
        '''found_verbs = []
                    for tag in pos_tagged:
                    	if tag in verb_tags:
                    		found_verbs.append(tag)
        found_verbs = [tag[0] for tag in pos_tagged if tag[1] in verb_tags]

        if len(found_verbs) == 1 and found_verbs[0] in ['is','are']: # these are attributes
            continue'''

        #how to handle when there are 2 or more verbs in a description?
        #srl doesn't work
        if parsed_phrase['hasVerb'] :
            print('Found verbs')
            srl_dict_list = srl(parsed_phrase['verbs'], objects)

            if srl_dict_list and len(srl_dict_list) > 0 :
                for element in srl_dict_list:
                    srl_dict = element['srl']
                    if srl_dict:
                        print ('current srl :', srl_dict)
                        role_list = OrderedDict()
                        elements_inside_region = 0
                        srl_found_elements = 0
                        current_matching_elements = []
                        print('tags', pos_tagged)
                        for tag in element['tags']:
                            print('current tag :', tag)
                            #depends on accuracy of written sentence and srl tool
                            # if srl hasn't identified something as a label, it is not considered for semantic map

                            if tag[1] in noun_tags:
                                matching_srl_label = [label for label in srl_dict.values() if tag[0] in label]

                                print('matching srl labels :', matching_srl_label)
                                if not len(matching_srl_label) > 0:
                                    continue
                                preposition = ''
                                #for prepositional phrases
                                if ',' in matching_srl_label[0]:
                                    split = matching_srl_label[0].split(',')
                                    preposition = split[0]
                                    matching_srl_label[0] = split[1]

                                if matching_srl_label[0] not in current_matching_elements:
                                    current_matching_elements.append(matching_srl_label[0])
                                    srl_found_elements += 1


                                for obj in objects:
                                    print('obj :', obj['names'][0], 'current element ',matching_srl_label[0])
                                    if obj['names'][0] == matching_srl_label[0]:
                                        print ('object: ', matching_srl_label[0], obj)
                                        obj_box = Rectangle(obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h'])
                                        # y should we calculate area like this?
                                        #obj_box_area = (obj_box.xmax - obj_box.xmin + 1) * (obj_box.ymax - obj_box.ymin + 1)
                                        object_overlap_ratio = get_verlap_ratio(region_box, obj_box, obj)
                                        print ('overlap ratio :', object_overlap_ratio)

                                        if object_overlap_ratio == 0.0:
                                    #otherwise this may lead to unrelated regions (some other's face mistaken as inside current region)
                                            continue

                                        #even with inimum overlap, consider the relations if we couldn't find a good one.
                                        # this is required because of incorrect region bounding boxes of human annotation
                                        if preposition:
                                            role = srl_dict.keys()[srl_dict.values().index(preposition + ',' + obj['names'][0])]
                                        else:
                                            role = srl_dict.keys()[srl_dict.values().index(obj['names'][0])]

                                        if preposition :
                                            if not role in role_list:
                                                elements_inside_region += 1
                                                role_list[role] = (object_overlap_ratio,preposition + ' ' + str(obj['object_id']))
                                            else:
                                                current_overlap = role_list[role][0]
                                                if current_overlap < object_overlap_ratio:
                                                    role_list[role] = (object_overlap_ratio,preposition + ' ' + str(obj['object_id']))
                                        else:
                                            if not role in role_list:
                                                elements_inside_region += 1
                                                role_list[role] = (object_overlap_ratio,obj['object_id'])
                                            else:
                                                current_overlap = role_list[role][0]
                                                if current_overlap < object_overlap_ratio:
                                                    role_list[role] = (object_overlap_ratio,obj['object_id'])

                        print('role list length', elements_inside_region, 'element lenght', srl_found_elements)
                        if elements_inside_region != srl_found_elements:
                            "removing this region from semantic maps"
                            continue

                        print('actual role list in the region ', role_list)
                        map_key = srl_dict['predicate']
                        if map_key in semantic_map:
                            semantic_map[map_key].append((role_list, region_id))
                            #what if existing roles have dfferent values?
                            '''diff = set(role_list.keys()) - set(roles.keys())
                            if diff:
                                for new_role in  role_list.keys() :
                                    roles[new_role] = role_list[new_role]
                                semantic_map[map_key] = roles'''
            			    # assuming that if same person is doing same action, there is no way he can do it in 2 different ways in an image
                        else :
                            semantic_map[map_key] = [(role_list, region_id)]

                print ('current semantic map : ', semantic_map)
    #todo: how to handle same entity identified in different names? person = girl
    print(semantic_map)
    '''
    clean dict, and group if for same predicate, same role, same entities
    then for each branch of the dict, create a relation entry in a relations list
    finally add image id to it
    and finish for current image and write to a file
    '''
    final_map = get_final_semantic_map(semantic_map)
    print('final map', final_map)
    return final_map

import sys
object_file_path = sys.argv[1]
region_file_path = sys.argv[2]
relationship_file_path = sys.argv[3]

# Main method of the program
def main():
    image_count = 0
    with open(object_file_path) as data_file1:
        content_objects = json.load(data_file1)
        # content_objects = json.load(data_file1,object_pairs_hook=OrderedDict)

    # content_image_regions = json.load(open(region_file_path), object_pairs_hook=OrderedDict)
    content_image_regions = json.load(open(region_file_path))

#    with open('/Users/thilinicooray/sem_img/sample/region_desc1edited.json') as data_file:
#        content_image_regions = json.load(data_file)
    with open (relationship_file_path, 'w') as relation_file:
        for image in content_image_regions:
            image_id = image['id']
            print('started processing image ', image_id)
            regions = image['regions']
            objects = filter(lambda x: x['image_id'] == image_id, content_objects)[0]['objects']

            final_semantic_map = get_semantic_map_for_image(regions, objects)

            if len(final_semantic_map) > 0:
                image_count += 1
                relation_entry = {'relationships' : final_semantic_map, 'image_id' : image_id}

                relation_file.write(json.dumps(relation_entry) + ', ')
                relation_file.flush()

                print('Successfully wrote  visual semantic map for image ', image_id, ' to output file')
                print('current dataset size = ' + str(image_count))

if __name__ == '__main__':
    main()
