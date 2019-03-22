#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:56:10 2019

@author: fredericvankelecom
"""

#Subclassifications 


#Notes:
#Valyrian house and group have a high mortality, consider joining them. 
#Ironborn and House Greyjoy as well, but much smaller. Try joining
#Age 100 and DOB 200 could be the same thing (60% pos corr), in which case drop
#the former.
#Undefined, nights watch and other have higher corr than the others
#Maybe try separating sirs and ladies in each house, to see if there is a greater corr
#Warriors and royals could give a better corr if further segmented

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',200)



file_real = 'GOT_character_predictions.xlsx'
got = pd.read_excel(file_real)


##############################################################################


# Column names
got.columns


# Displaying the first rows of the DataFrame
print(got.head())


# Dimensions of the DataFrame
got.shape


# Information about each variable
got.info()


# Descriptive statistics
got.describe().round(2)


got.sort_values('isAlive', ascending = False)

###############################################################################

# flagging Missing Values

###############################################################################

print(
      got
      .isnull()
      .sum()
      )

for col in got:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)
        
###############################################################################

# Make everything lower case

###############################################################################
 
#Make everything that is an object lower case       
for col in got.columns:
  if got[col].dtype == object:
     got[col] = got[col].str.lower()
     

###############################################################################

# 0. Correcting genders

###############################################################################


male_list = ['abelar',
 'addam',
 'addison',
 'aegon',
 'aegor',
 'aemon',
 'aenar',
 'aenys',
 'aeron',
 'aerys',
 'aethelmure',
 'aggar',
 'aggo',
 'agrivane',
 'aladale',
 'albar',
 'albett',
 'alebelly',
 'alequo',
 'alesander',
 'alester',
 'alfyn',
 'allar',
 'allard',
 'alleras',
 'alliser',
 'alvyn',
 'alyce',
 'alyn',
 'amabel',
 'ambrode',
 'ambrose',
 'amerei',
 'amory',
 'andar',
 'anders',
 'andrew',
 'andrey',
 'andrik',
 'andros',
 'androw',
 'anguy',
 'antario',
 'anvil',
 'ardrian',
 'areo',
 'argilac',
 'argrave',
 'arlan',
 'armen',
 'armond',
 'arneld',
 'arnell',
 'arnolf',
 'aron',
 'arrec',
 'arron',
 'arryk',
 'arson',
 'arthur',
 'artos',
 'artys',
 'arwood',
 'arwyn',
 'arys',
 'assadora',
 'aubrey',
 'aurane',
 'axell',
 'bael',
 'baelor',
 'ballabar',
 'balman',
 'balon',
 'bandy',
 'bannen',
 'barristan',
 'barth',
 'bass',
 'bayard',
 'beardless',
 'beck',
 'bedwyck',
 'belgrave',
 'bellegere',
 'belwas',
 'ben',
 'benedar',
 'benedict',
 'benfred',
 'benfrey',
 'benjen',
 'bennard',
 'bennarion',
 'bennet',
 'bennis',
 'beren',
 'beric',
 'bernarr',
 'beron',
 'bertram',
 'betharios',
 'big\xa0boil',
 'biter',
 'black',
 'blane',
 'bluetooth',
 'bodger',
 'bonifer',
 'borcas',
 'boremund',
 'boros',
 'bors',
 'bowen',
 'bradamar',
 'bran',
 'brandon',
 'branston',
 'brenett',
 'bronn',
 'brusco',
 'bryan',
 'bryce',
 'bryen',
 'brynden',
 'buford',
 'burton',
 'butterbumps',
 'byam',
 'byren',
 'byron',
 'cadwyl',
 'cadwyn',
 'caleotte',
 'calon',
 'carrot',
 'castos',
 'cayn',
 'cedric',
 'cedrik',
 'cellador',
 'cerrick',
 'cetheres',
 'chayle',
 'chett',
 'chiggen',
 'chiswyck',
 'clarence',
 'clayton',
 'clement',
 'cleon',
 'cleos',
 'cletus',
 'cley',
 'cleyton',
 'clifford',
 'clydas',
 'cohollo',
 'coldhands',
 'colemon',
 'colen',
 'colin',
 'collio',
 'colloquo',
 'colmar',
 'conn',
 'conwy',
 'corliss',
 'cortnay',
 'cosgrove',
 'cotter',
 'courtenay',
 'cragorn',
 'craster',
 'crawn',
 'cregan',
 'creighton',
 'cressen',
 'creylen',
 'criston',
 'cuger',
 'cutjack',
 'daario',
 'dacks',
 'daemon',
 'daeron',
 'dafyn',
 'dagmer',
 'dagon',
 'dagos',
 'dake',
 'dale',
 'damion',
 'damon',
 'dan',
 'dancy',
 'danny',
 'danwell',
 'dareon',
 'daryn',
 'daven',
 'davos',
 'del',
 'delonne',
 'delp',
 'denestan',
 'dennett',
 'dennis',
 'denyo',
 'denys',
 'deremond',
 'dermot',
 'desmond',
 'devan',
 'deziel',
 'dick',
 'dickon',
 'dilly',
 'dirk',
 'dobber',
 'dolf',
 'domeric',
 'donal',
 'donel',
 'donella',
 'donnel',
 'donnis',
 'donnor',
 'doran',
 'dormund',
 'dorren',
 'drennan',
 'dryn',
 'dudley',
 'dunaver',
 'dunk',
 'dunsen',
 'dunstan',
 'duram',
 'durran',
 'dyah',
 'dykk',
 'ebrose',
 'eddard',
 'edderion',
 'eddison',
 'eden',
 'edgerran',
 'edmund',
 'edmure',
 'edmyn',
 'edric',
 'edrick',
 'edwyd',
 'edwyle',
 'edwyn',
 'eggon',
 'elbert',
 'eldiss',
 'eldon',
 'eldred',
 'ellery',
 'elmar',
 'elwood',
 'elyas',
 'elyn',
 'elys',
 'emmon',
 'emmond',
 'emrick',
 'endehar',
 'endrew',
 'enger',
 'eon',
 'erik',
 'eroeh',
 'erreck',
 'erreg',
 'erren',
 'errok',
 'erryk',
 'esgred',
 'ethan',
 'euron',
 'eustace',
 'eyron',
 'farlen',
 'ferrego',
 'ferret',
 'flement',
 'fogo',
 'forley',
 'fralegg',
 'franklyn',
 'frenken',
 'gage',
 'galbart',
 'galladon',
 'gallard',
 'galtry',
 'garin',
 'gariss',
 'garizon',
 'garlan',
 'garrett',
 'garrison',
 'garse',
 'garth',
 'gascoyne',
 'gawen',
 'gelmarr',
 'gendel',
 'gendry',
 'gerald',
 'gerardys',
 'geremy',
 'gergen',
 'gerion',
 'germund',
 'gerold',
 'gerren',
 'gevin',
 'ghael',
 'ghost',
 'gilbert',
 'gillam',
 'gilwood',
 'gladden',
 'glendon',
 'godric',
 'godry',
 'godwyn',
 'goodwin',
 'gorghan',
 'gormon',
 'gormond',
 'gorne',
 'gorold',
 'gowen',
 'gran',
 'grance',
 'grazdan',
 'greatheart',
 'greenbeard',
 'gregor',
 'grenn',
 'grey',
 'greydon',
 'griffin',
 'grigg',
 'groleo',
 'grubbs',
 'gueren',
 'gulian',
 'gunthor',
 'guthor',
 'guyard',
 'guyne',
 'gwayne',
 'gylbert',
 'gyles',
 'gynir',
 'haegon',
 'haereg',
 'haggo',
 'hake',
 'hal',
 'halder',
 'halleck',
 'hallis',
 'halmon',
 'halys',
 'harbert',
 'hareth',
 'harlan',
 'harlen',
 'harlon',
 'harmen',
 'harmond',
 'harmund',
 'harmune',
 'harodon',
 'harrag',
 'harras',
 'harren',
 'harrion',
 'harrold',
 'harry',
 'harsley',
 'harwin',
 'harwood',
 'harwyn',
 'harys',
 'hayhead',
 'helicent',
 'helliweg',
 'helman',
 'hendry',
 'henk',
 'henly',
 'herbert',
 'heward',
 'hibald',
 'hilmar',
 'hobb',
 'hobber',
 'hod',
 'hodor',
 'hoke',
 'hop-robin',
 'horas',
 'horton',
 'hosman',
 'hosteen',
 'hoster',
 'hot',
 'hother',
 'hotho',
 'howland',
 'hugh',
 'hugo',
 'hullen',
 'humfrey',
 'hyle',
 'iggo',
 'igon',
 'illifer',
 'illyrio',
 'ilyn',
 'imry',
 'ironbelly',
 'jacelyn',
 'jack',
 'jack-be-lucky',
 'jacks',
 'jaehaerys',
 'jafer',
 'jaggot',
 'jaime',
 'jammos',
 'janos',
 'jaqen',
 'jared',
 'jarl',
 'jason',
 'jasper',
 'jate',
 'jeffory',
 'jeor',
 'jeren',
 'jhaqo',
 'jhogo',
 'jodge',
 'joffrey',
 'jojen',
 'jommo',
 'jon',
 'jonnel',
 'jonos',
 'jonothor',
 'jorah',
 'joramun',
 'jorgen',
 'jorquen',
 'jory',
 'joseran',
 'joseth',
 'josmyn',
 'joss',
 'josua',
 'joth',
 'joy',
 'jurne',
 'jyck',
 'kaeth',
 'karl',
 'karlon',
 'karyl',
 'kedge',
 'kedry',
 'kemmett',
 'kenned',
 'kennos',
 'ketter',
 'kevan',
 'khal',
 'khorane',
 'kirby',
 'kirth',
 'koss',
 'kraznys',
 'kromm',
 'kurleket',
 'kurz',
 'kyle',
 'kym',
 'lambert',
 'lancel',
 'larence',
 'lark',
 'lazy',
 'lem',
 'lenn',
 'lennocks',
 'lenwood',
 'lenyl',
 'leo',
 'leobald',
 'lester',
 'lew',
 'lewyn',
 'lewys',
 'leyton',
 'lharys',
 'lister',
 'lomas',
 'lommy',
 'lomys',
 'loras',
 'lorcas',
 'lord',
 'loren',
 'lorent',
 'lorimer',
 'lormelle',
 'lorren',
 'lothar',
 'lothor',
 'lucamore',
 'lucan',
 'lucantine',
 'lucas',
 'luceon',
 'lucias',
 'lucimore',
 'lucion',
 'luco',
 'lucos',
 'luke',
 'lum',
 'luthor',
 'luwin',
 'lyle',
 'lyman',
 'lymond',
 'lyn',
 'lync',
 'lyonel',
 'mace',
 'mad',
 'maegor',
 'maekar',
 'maelys',
 'mag',
 'mago',
 'mallador',
 'malleon',
 'mallor',
 'malwyn',
 'mance',
 'mandon',
 'manfred',
 'manfrey',
 'manfryd',
 'maric',
 'marillion',
 'mark',
 'marlon',
 'maron',
 'marq',
 'martyn',
 'marwyn',
 'maslyn',
 'mathis',
 'matt',
 'matthar',
 'matthos',
 'maynard',
 'mebble',
 'medgar',
 'medger',
 'medwick',
 'meizo',
 'melaquin',
 'meldred',
 'mellario',
 'melwyn',
 'melwys',
 'meribald',
 'merlon',
 'mern',
 'mero',
 'merrell',
 'merrett',
 'merrit',
 'meryn',
 'michael',
 'mikken',
 'mohor',
 'mollos',
 'moon',
 'mord',
 'mordane',
 'moreo',
 'morgan',
 'morgarth',
 'morgil',
 'moribald',
 'moro',
 'morosh',
 'morrec',
 'mors',
 'mortimer',
 'morton',
 'moryn',
 'mudge',
 'mullin',
 'munda',
 'murch',
 'murenmure',
 'mycah',
 'mychel',
 'myles',
 'nage',
 'naggle',
 'narbert',
 'ned',
 'nestor',
 "night's",
 'noho',
 'norbert',
 'norjen',
 'normund',
 'norne',
 'norren',
 'notch',
 'nute',
 'nymos',
 'oberyn',
 'ocley',
 'ogo',
 'old',
 'ollidor',
 'ollo',
 'olymer',
 'olyvar',
 'omer',
 'ondrew',
 'orbelo',
 'orbert',
 'orell',
 'orivel',
 'orland',
 'ormond',
 'oro',
 'orton',
 'orys',
 'osbert',
 'osmund',
 'osmynd',
 'osric',
 'ossifer',
 'ossy',
 'oswell',
 'othell',
 'otho',
 'othor',
 'otter',
 'otto',
 'ottomore',
 'ottyn',
 'owen',
 'parmen',
 'patchface',
 'pate',
 'patrek',
 'paxter',
 'pearse',
 'perestan',
 'perros',
 'perwyn',
 'peter',
 'petyr',
 'philip',
 'plummer',
 'podrick',
 'polliver',
 'pono',
 'porther',
 'portifer',
 'poxy',
 'praed',
 'prendahl',
 'preston',
 'puckens',
 'pudding',
 'puddingfoot',
 'pycelle',
 'pylos',
 'pypar',
 'qalen',
 'qarl',
 'qarlton',
 'qarro',
 'qhored',
 'qos',
 'qotho',
 'quaithe',
 'quaro',
 'quellon',
 'quent',
 'quenten',
 'quentin',
 'quenton',
 'quentyn',
 'quincy',
 'quort',
 'qyburn',
 'qyle',
 "r'hllor",
 'rafe',
 'rafford',
 'ragnor',
 'ragwyle',
 'rakharo',
 'ralf',
 'ramsay',
 'randyll',
 'rast',
 'rat',
 'rattleshirt',
 'rawney',
 'raymar',
 'raymond',
 'raymun',
 'raymund',
 'raynald',
 'raynard',
 'red',
 'redtusk',
 'reek',
 'regenard',
 'renly',
 'reynard',
 'reysen',
 'rhaegar',
 'rhaego',
 'rhogoro',
 'ricasso',
 'richard',
 'rickard',
 'rickon',
 'rigney',
 'rob',
 'robar',
 'robb',
 'robert',
 'robett',
 'robin',
 'robyn',
 'rodrik',
 'rodwell',
 'roger',
 'roland',
 'rolder',
 'rolfe',
 'rollam',
 'rolland',
 'rolley',
 'rolph',
 'romny',
 'ronald',
 'ronel',
 'ronnel',
 'ronnet',
 'roone',
 'roose',
 'rorge',
 'roro',
 'roryn',
 'rowena',
 'royce',
 'rudge',
 'rufus',
 'runceford',
 'runcel',
 'rupert',
 'rus',
 'russell',
 'ryam',
 'rycherd',
 'ryger',
 'ryk',
 'ryman',
 'rymolf',
 'ryon',
 'salladhor',
 'salloreon',
 'sam',
 'samwell',
 'sand',
 'sandor',
 'sargon',
 'satin',
 'sawane',
 'sawwood',
 'sebaston',
 'sedgekins',
 'selmond',
 'selwyn',
 'serwyn',
 'shadd',
 'shadrich',
 'shagga',
 'shagwell',
 'sherrit',
 'sigfry',
 'sigfryd',
 'sigorn',
 'sigrin',
 'simon',
 'skittrick',
 'skyte',
 'sleepy',
 'smiling',
 'softfoot',
 'spotted',
 'squint',
 'stafford',
 'stannis',
 'steffarion',
 'steffon',
 'stevron',
 'stiv',
 'stone',
 'stonehand',
 'stygg',
 'styr',
 'sumner',
 'sylas',
 'symon',
 'symond',
 'syrio',
 'talbert',
 'tallad',
 'tanton',
 'tarle',
 'temmo',
 'ternesio',
 'terrance',
 'terrence',
 'terro',
 'theo',
 'theobald',
 'theodan',
 'theodore',
 'theomar',
 'theomore',
 'theon',
 'thomax',
 'thoren',
 'thormor',
 'three-tooth',
 'tickler',
 'tim',
 'timeon',
 'timett',
 'timon',
 'timoth',
 'tion',
 'titus',
 'tobbot',
 'tobho',
 'todder',
 'todric',
 'toefinger',
 'tom',
 'tomtoo',
 'tomard',
 'tommard',
 'tommen',
 'toregg',
 'tormund',
 'torrek',
 'torren',
 'torrhen',
 'torwold',
 'torwynd',
 'tothmure',
 'trebor',
 'tregar',
 'tremond',
 'tristan',
 'tristifer',
 'tristimun',
 'triston',
 'trystane',
 'tuffleberry',
 'turnip',
 'turquin',
 'tybolt',
 'tygett',
 'tymor',
 'tyrek',
 'tyrion',
 'tytos',
 'tywin',
 'ulf',
 'uller',
 'ulrick',
 'ulwyck',
 'umfred',
 'urek',
 'urras',
 'urrigon',
 'urron',
 'urzen',
 'utherydes',
 'uthor',
 'vaellyn',
 'varamyr',
 'vardis',
 'vargo',
 'varly',
 'varys',
 'vayon',
 'vickon',
 'victarion',
 'victor',
 'viserys',
 'vortimer',
 'vylarr',
 'vyman',
 'walder',
 'waldon',
 'walgrave',
 'wallace',
 'wallen',
 'walton',
 'waltyr',
 'warren',
 'warryn',
 'wat',
 'watt',
 'watty',
 'waymar',
 'wayn',
 'weeper',
 'weese',
 'wendamyr',
 'wendel',
 'wendell',
 'werlag',
 'wex',
 'whalen',
 'wilbert',
 'will',
 'willam',
 'willamen',
 'willas',
 'willem',
 'william',
 'willifer',
 'willis',
 'willit',
 'willum',
 'wolmer',
 'wulfe',
 'wyl',
 'wylis',
 'wyman',
 'wynton',
 'ygon',
 'yohn',
 'yoren',
 'yorko',
 'yormwell',
 'zachery',
 'zarabelo',
 'zollo']

female_list = ['alannys',
 'alayaya',
 'alekyne',
 'alerie',
 'alia',
 'alla',
 'allyria',
 'alyce',
 'alys',
 'alysane',
 'alysanne',
 'alyse',
 'alyssa',
 'alyx',
 'amabel',
 'amarei',
 'anya',
 'arianne',
 'arwyn',
 'arya',
 'asha',
 'ashara',
 'assadora',
 'barba',
 'barbrey',
 'barra',
 'becca',
 'belandra',
 'bella',
 'bellena',
 'bellonara',
 'beony',
 'berena',
 'bessa',
 'beth',
 'bethany',
 'brella',
 'brienne',
 'canker',
 'carellen',
 'carolei',
 'cass',
 'cassana',
 'cassella',
 'catelyn',
 'cedra',
 'cerenna',
 'cersei',
 'chella',
 'corenna',
 'cynthea',
 'cyrenna',
 'dacey',
 'daenerys',
 'dalla',
 'danelle',
 'darlessa',
 'deana',
 'delena',
 'della',
 'denyse',
 'desmera',
 'dorcas',
 'dorea',
 'doreah',
 'dorna',
 'dyah',
 'eddara',
 'eleanor',
 'elenei',
 'elenya',
 'elia',
 'elinor',
 'ellaria',
 'ellyn',
 'elyana',
 'emberlei',
 'emma',
 'emphyria',
 'erena',
 'ermesande',
 'fair',
 'falia',
 'fern',
 'ferny',
 'frynne',
 'gilly',
 'gretchel',
 'grisel',
 'gwin',
 'gwynesse',
 'gysella',
 'hali',
 'harma',
 'harra',
 'helly',
 'helya',
 'hostella',
 'irri',
 'janei',
 'janna',
 'janyce',
 'jayde',
 'jennelyn',
 'jenny',
 'jeyne',
 'jhiqui',
 'joanna',
 'jocelyn',
 'jonella',
 'jorelle',
 'joy',
 'joyeuse',
 'jyana',
 'jyanna',
 'jynessa',
 'jyzene',
 'kella',
 'kyra',
 'lady',
 'lanna',
 'larra',
 'layna',
 'leana',
 'leona',
 'leonella',
 'leonette',
 'leslyn',
 'leyla',
 'lia',
 'liane',
 'loreza',
 'lyanna',
 'lyessa',
 'lynesse',
 'lyra',
 'lysa',
 'lythene',
 'maddy',
 'maege',
 'maegelle',
 'maerie',
 'maggy',
 'malora',
 'margaery',
 'margot',
 'marianne',
 'marissa',
 'mariya',
 'marsella',
 'marya',
 'masha',
 'matrice',
 'meera',
 'meg',
 'megga',
 'mela',
 'melara',
 'melesa',
 'melessa',
 'meliana',
 'melisandre',
 'mellara',
 'mellei',
 'melly',
 'meralyn',
 'meredyth',
 'merianne',
 'mhaegan',
 'mina',
 'minisa',
 'mirri',
 'missandei',
 'moonshadow',
 'morra',
 'morya',
 'munda',
 'mya',
 'mylenda',
 'mylessa',
 'myranda',
 'myrcella',
 'myria',
 'myriah',
 'myrielle',
 'naerys',
 'nan',
 'nella',
 'nightingale',
 'nolla',
 'nymella',
 'nymeria',
 'obara',
 'obella',
 'olene',
 'olenna',
 'osha',
 'palla',
 'perra',
 'perriane',
 'pia',
 'randa',
 'ravella',
 'rhaenys',
 'rhea',
 'rhialta',
 'rhonda',
 'roelle',
 'rohanne',
 'rosamund',
 'rosey',
 'roslin',
 'ryella',
 'rylene',
 'sallei',
 'sansa',
 'sarella',
 'sarra',
 'sarya',
 'selyse',
 'senelle',
 'serala',
 'serenei',
 'serra',
 'shae',
 'sharna',
 'shella',
 'shiera',
 'shierle',
 'shirei',
 'shyra',
 'sybell',
 'sybelle',
 'sylva',
 'sylwa',
 'taena',
 'tanselle',
 'tansy',
 'tya',
 'tyana',
 'tyene',
 'tysane',
 'tysha',
 'tyta',
 'umma',
 'val',
 'victaria',
 'violet',
 'walda',
 'weasel',
 'white',
 'willow',
 'wylla',
 'wynafrei',
 'wynafryd',
 'ygritte',
 'yna',
 'ysilla',
 'zei',
 'zhoe',
 'zia']


names = got.name.str.split(expand=True,)
first_names = names.iloc[:,0]

got['first_names'] = first_names

got['male_copy'] = got['male']


for i in range(0,len(got)):
    if got.loc[i,'first_names'] in list(male_list):
       got.loc[i,'male_copy'] = 1

for i in range(0,len(got)):
    if got.loc[i,'first_names'] in list(female_list):
       got.loc[i,'male_copy'] = 0


print(
      got
      .isnull()
      .sum()
      )
     
###############################################################################

# 1. Title

###############################################################################

got.title.nunique()

#Creating Lords and Ladies
got['title_filled'] = got['title'].fillna('undef_title')

list1 = ['Stonehelm',
         'Harrenhal',
         'Coldmoat',
         'Gulltown',
         'Longtable',
         'Dreadfort',
         'Yronwood',
         'Acorn Hall',
         'Riverrun',
         'Winterfell',
         'Nightsong',
         'Starpike',
         'Sweetport Sound',
         'the Dreadfort',
         'Bear Island',
         'Ruddy Hall',
         'Sandstone',
         'Raventree Hall',
         'Arbor',
         'Eyrie',
         'Karhold',
         'Longsister',
         "Storm's End",
         'Casterly Rock',
         'Deepwood Motte',
         'Strongsong', 
         'Highgarden',
         'Felwood',
         'Ashford',
         'Golden Storm',
         'Stone Hedge',
         'Rain House',
         'Goldengrove',
         'Blacktyde',
         'Ten Towers',
         'Duskendale',
         'Blackcrown',
         'Godsgrace',
         'Redfort',
         "Karl's Hold",
         'Fair Isle',
         'the Crossing',
         'Bitterbridge',
         'Runestone',
         'Wraith',
         'Darry',
         'Stokeworth',
         'Hornwood',
         'Rills',
         'Old Wyk',
         'Sandship',
         'Wyndhall',
         'Twins',
         'Hightower',
         'Dragonstone',
         'Kayce',
         'Iron Islands',
         'Broad Arch',
         'Last Hearth', 
         'Eastwatch-by-the-Sea',
         'Crakehall',
         'Sunspear',
         'Greywater Watch',
         'Sweetsister',
         'Shatterstone',
         'Three Towers',
         'Seagard',
         'Blackmont',
         'Tower of Glimmering',
         'Feastfires',
         'Ironoaks',
         'Longbow Hall',
         'Barrowton',
         'Banefort',
         'Crag',
         'Brightwater',
         'Cerwyn',
         'Ghost hill',
         'greenstone',
         'volmark',
         'harlow',
         'maidenpool',
         "widow's watch",
         "rook's rest",
         'goldgrass',
         'vaith',
         'hornvale',
         'greenshield',
         'skyreach',
         'old oak',
         'claw isle',
         'red flower vale',
         'whitewalls',
         "heart's home",
         'horn hill',
         'oakenshield',
         'uplands',
         'hayford',
         'dyre den',
         'salt shore',
         'three sisters',
         'grey glen',
         'lonely light',
         'harridan hill']

#Make the list lower case

list1 = [x.lower() for x in list1]


for i in range(0, len(got['title_filled'])):
    if (got.loc[i, 'title_filled'] in list1 and got.loc[i, 'male'] == 1):
        got['title_filled'][i] = 'lord of ' + got['title_filled'][i]
    elif (got.loc[i, 'title_filled'] in list1 and got.loc[i, 'male'] == 0):
       got['title_filled'][i] = 'lady of ' + got['title_filled'][i]
       
     
#Change title to lord or lady
got.loc[got['title_filled'].str.contains('lord'), 'title_filled'] = 'lord'
got.loc[got['title_filled'].str.contains('lady'), 'title_filled'] = 'lady'
got.loc[got['title_filled'].str.contains('knight'), 'title_filled'] = 'ser'

got['title_filled'].value_counts()

#group the titles
got['title_filled_grouped'] = 'other'

#No Title
got.loc[got['title_filled'].str.contains('undef_title'), 'title_filled_grouped'] = 'no_title'

#Lord/Lady
got.loc[got['title_filled'].str.contains('lord'), 'title_filled_grouped'] = 'lord'

got.loc[got['title_filled'].str.contains('lady'), 'title_filled_grouped'] = 'lady'

#Royals
got.loc[got['title_filled'].str.contains('princess'), 'title_filled_grouped'] = 'royals'
got.loc[got['title_filled'].str.contains('prince'), 'title_filled_grouped'] = 'royals'
got.loc[got['title_filled'].str.contains('king'), 'title_filled_grouped'] = 'king'

#got.loc[got['title_filled'].str.contains('khal'), 'title_filled_grouped'] = 'royals'
got.loc[got['title_filled'].str.contains('princessqueen'), 'title_filled_grouped'] = 'royals'
#got.loc[got['title_filled'].str.contains('khalko (formerly)'), 'title_filled_grouped'] = 'royals'
got.loc[got['title_filled'].str.contains('queen'), 'title_filled_grouped'] = 'royals'

#Clergymen
got.loc[got['title_filled'].str.contains('maester'), 'title_filled_grouped'] = 'clergymen'
got.loc[got['title_filled'].str.contains('archmaester'), 'title_filled_grouped'] = 'clergymen'
#got.loc[got['title_filled'].str.contains('septon'), 'title_filled_grouped'] = 'clergymen'
#got.loc[got['title_filled'].str.contains('septa'), 'title_filled_grouped'] = 'clergymen'
got.loc[got['title_filled'].str.contains('brother'), 'title_filled_grouped'] = 'clergymen'
got.loc[got['title_filled'].str.contains('wisdom'), 'title_filled_grouped'] = 'warrior'

#Warriors
got.loc[got['title_filled'].str.contains('ser'), 'title_filled_grouped'] = 'noble_warrior'
got.loc[got['title_filled'].str.contains('bloodrider'), 'title_filled_grouped'] = 'warrior'
got.loc[got['title_filled'].str.contains('first ranger'), 'title_filled_grouped'] = 'warrior'
got.loc[got['title_filled'].str.contains('captain'), 'title_filled_grouped'] = 'noble_warrior'


#Check others
df_others = got[got['title_filled_grouped'] == 'other']
df_others['title_filled'].value_counts()

###############################################################################

# 3. Houses

###############################################################################

# Combining the different Cadets of Houses
got['house_filled'] = got['house'].fillna('undef_house')

got.loc[got['house_filled'].str.contains('house bolton'), 'house_filled'] = 'house bolton'

got.loc[got['house_filled'].str.contains('house dayne'), 'house_filled'] = 'house dayne'

got.loc[got['house_filled'].str.contains('house farwynd'), 'house_filled'] = 'house farwynd'

got.loc[got['house_filled'].str.contains('house flint'), 'house_filled'] = 'house flint'

got.loc[got['house_filled'].str.contains('house fossoway'), 'house_filled'] = 'house fossoway'

got.loc[got['house_filled'].str.contains('house fray'), 'house_filled'] = 'house fray'

got.loc[got['house_filled'].str.contains('house goodbrother'), 'house_filled'] = 'house goodbrother'

got.loc[got['house_filled'].str.contains('house harlaw'), 'house_filled'] = 'house harlaw'

got.loc[got['house_filled'].str.contains('house lannister'), 'house_filled'] = 'house lannister'

got.loc[got['house_filled'].str.contains('house royce'), 'house_filled'] = 'house royce'

got.loc[got['house_filled'].str.contains('house tyrell'), 'house_filled'] = 'house tyrell'

got.loc[got['house_filled'].str.contains('citadel'), 'house_filled'] = 'citadel'

got.loc[got['house_filled'].str.contains('baratheon'), 'house_filled'] = 'house baratheon'

###############################################################################

# 4. Correct assignment of character to house

###############################################################################

got.loc[got['name'].str.contains('targaryen'), 'house_filled'] = 'house targaryen'

got.loc[got['name'].str.contains('stark'), 'house_filled'] = 'house stark'

got.loc[got['name'].str.contains('lannister'), 'house_filled'] = 'house lannister'

got.loc[got['name'].str.contains('frey'), 'house_filled'] = 'house frey'

got.loc[got['name'].str.contains('greyjoy'), 'house_filled'] = 'house greyjoy'

got.loc[got['name'].str.contains('martell'), 'house_filled'] = 'house martell'

got.loc[got['name'].str.contains('osgrey'), 'house_filled'] = 'house osgrey'

got.loc[got['name'].str.contains('arryn'), 'house_filled'] = 'house arryn'

got.loc[got['name'].str.contains('hightower'), 'house_filled'] = 'house hightower'

#Addition
got.loc[got['name'].str.contains('oldtown'), 'house_filled'] = 'house hightower'
###############################################################################

got.loc[got['name'].str.contains('royce'), 'house_filled'] = 'house royce'

got.loc[got['name'].str.contains('bolton'), 'house_filled'] = 'house bolton'

got.loc[got['name'].str.contains('bracken'), 'house_filled'] = 'house bracken'

got.loc[got['name'].str.contains('florent'), 'house_filled'] = 'house florent'

got.loc[got['name'].str.contains('botley'), 'house_filled'] = 'house botley'

got.loc[got['name'].str.contains('baratheon'), 'house_filled'] = 'house baratheon'

got.loc[got['name'].str.contains('tully'), 'house_filled'] = 'house tully'

got.loc[got['name'].str.contains('whent'), 'house_filled'] = 'house whent'

got.loc[got['name'].str.contains('velaryon'), 'house_filled'] = 'house velaryon'

got.loc[got['name'].str.contains('crakehall'), 'house_filled'] = 'house crakehall'

got.loc[got['name'].str.contains('harlaw'), 'house_filled'] = 'house harlaw'

#Cut off point houses: from and exclusive 5 houses 
df_house = got['house_filled'].value_counts()
df_house = df_house.reset_index()

df_house['house_fraction'] = 0

list_small_houses = []

for i in range(0,len(df_house)):
    if df_house.loc[i,'house_filled']<=4:
        list_small_houses.append(df_house.loc[i,'index'])
        
got['house_complete'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list_small_houses:
       got.loc[i,'house_complete'] = 'other_house'
    else:
       got.loc[i,'house_complete'] = got.loc[i,'house_filled']
       
got.loc[:,['house_filled', 'house_complete']]

got.house_complete.nunique()


###############################################################################

# 4. Culture

###############################################################################

got['culture_filled'] = got['culture'].fillna('undef_culture')

got.loc[got['culture_filled'].str.contains('braavos'), 'culture_filled'] = 'braavosi'
got.loc[got['culture_filled'].str.contains('wildling'), 'culture_filled'] = 'wildlings'
got.loc[got['culture_filled'].str.contains('lyseni'), 'culture_filled'] = 'lysene'
got.loc[got['culture_filled'].str.contains('westerman'), 'culture_filled'] = 'westermen'
got.loc[got['culture_filled'].str.contains('andal'), 'culture_filled'] = 'andals'
got.loc[got['culture_filled'].str.contains('meereenese'), 'culture_filled'] = 'meereen'
got.loc[got['culture_filled'].str.contains('asshai'), 'culture_filled'] = 'asshai'
got.loc[got['culture_filled'].str.contains('summer'), 'culture_filled'] = 'summer isles'
got.loc[got['culture_filled'].str.contains('lhazarene'), 'culture_filled'] = 'lhazareen'
got.loc[got['culture_filled'].str.contains('norvos'), 'culture_filled'] = 'norvoshi'
got.loc[got['culture_filled'].str.contains('stormlander'), 'culture_filled'] = 'stormlands'
got.loc[got['culture_filled'].str.contains('ironmen'), 'culture_filled'] = 'ironborn'
got.loc[got['culture_filled'].str.contains('astapor'), 'culture_filled'] = 'astapori'
got.loc[got['culture_filled'].str.contains('reachmen'), 'culture_filled'] = 'reach'
got.loc[got['culture_filled'].str.contains('the reach'), 'culture_filled'] = 'reach'
got.loc[got['culture_filled'].str.contains('ghiscaricari'), 'culture_filled'] = 'ghiscari'
got.loc[got['culture_filled'].str.contains('vale'), 'culture_filled'] = 'vale mountain clans'
got.loc[got['culture_filled'].str.contains('qarth'), 'culture_filled'] = 'qartheen'
got.loc[got['culture_filled'].str.contains('qarth'), 'culture_filled'] = 'qartheen'
got.loc[got['culture_filled'].str.contains('dorne'), 'culture_filled'] = 'dornish'
got.loc[got['culture_filled'].str.contains('dornishmen'), 'culture_filled'] = 'dornish'
got.loc[got['culture_filled'].str.contains('westermen'), 'culture_filled'] = 'westerlands'
got.loc[got['culture_filled'].str.contains('rivermen'), 'culture_filled'] = 'riverlands'



got['culture_filled'].value_counts()


###############################################################################

# 5. Allegiance

###############################################################################

house_stark = ['house reed',
             'house stark',
             'house glover',
             'house mormont',
             'house umber',
             'house manderly',
             'house flint',
             'house wull',
             'house norrey',
             'house liddle']
#             'house tully',
 #            'house westerling']
               

house_lannister = ['house lannister',		
             'house algood',
             'house banefort',
             'house brax',
             'house broom',
             'house clegane',
             'house crake hall',
             'house estren',
             'house farman',
             'house kenning',
             'house lefford',
             'house lorch',
             'house lydden',
             'house marbrand',
             'house payne',
             'house reyne',
             'house prester',
             'house serrett',
             'house spicer',
             'house swyft',
             'house tarbeck',
             'house westerling'] 


house_arryn = ['house arryn',		
             'house baelish',
             'house belmore',
             'house corbray',
             'house donniger',
             'house egen',
             'house elesham',
             'house grafton',
             'house hersy',
             'house hunter',
             'house lynderly',
             'house melcolm',
             'house moore',
             'house pryor',
             'house redfort',
             'house royce',
             'house ruthermont',
             'house sunderland',
             'house templeton',
             'house upcliff',
             'house waynwood',
             'house waxley']


house_tully = ['house tully',
               'House Blackwood',
               'House Bracken',
               'House Stone Hedge',
               'House Darry',
#               'House Frey',
               'House of the Crossing',
               'House Mallister',
               'House Mooton',
               'House Maidenpool']


house_greyjoy = ['House Blacktyde',
                 'house greyjoy',
                'House Botley',
                'House Codd',
                'House Drumm',
                'House Farwynd',
                'House Goodbrother',
                'House Hammerhorn',
                'House Corpse Lake',
                'House Crow Spike Keep',
                'House Downdelving',
                'House Shatterstone',
                'House Harlaw',
                'House Ten Towers',
                'House Grey Garden',
                'House the Tower of Glimmering',
                'House Harridan Hill',
                'House Humble',
                'House Ironmaker',
                'House Kenning',
                'House Merlyn',
                'House Myre',
                'House Netley',
                'House Orkwood ',
                'House Orkmont',
                'House Saltcliffe',
                'House Sharp',
                'House Shepherd',
                'House Sparr',
                'House Great Wyk',
                'House Stonehouse ',
                'House Old Wyk',
                'House Stonetree',
                'House Sunderly',
                'House Tawney',
                'House Volmark',
                'House Weaver',
                'House Wynch',
                'House Iron Holt']
                                
                
house_baratheon = ['House Bolling',
                   'house baratheon',
                    'House Buckler',
                    'House Cafferen',
                    'House Caron',
                    'House Connington',
                    'House Dondarrion',
                    'House Errol',
                    'House Estermont',
                    'House Fell',
                    'House Gower',
                    'House Grandison',
                    'House Hasty',
                    'House Herston',
                    'House Horpe',
                    'House Kellington',
                    'House Lonmouth',
                    'House Meadows',
                    'House Mertyns',
                    'House Morrigen',
                    'House Musgood',
                    'House Peasebury',
                    'House Penrose',
                    'House Rogers',
                    'House Selmy',
                    'House Staedmon',
                    'House Swann ',
                    'House Swygert',
                    'House Tarth ',
                    'House Trant ',
                    'House Tudbury',
                    'House Wagstaff',
                    'House Wensington',
                    'House Wylde']

house_tyrell = ['House Ambrose',
                'house tyrell',
                'House Appleton',
                'House Ashford',
                'House Ball',
                'House Blackbar',
                'House Bridges',
                'House Bushy',
                'House Caswell',
                'House Chester',
                'House Cockshaw',
                'House Conklyn',
                'House Cordwayner',
                'House Crane',
                'House Dunn',
                'House Durwell',
                'House Florent',
                'House Footly',
                'House Fossoway',
                'House Graceford',
                'House Graves',
                'House Grimm',
                'House Hastwyck',
                'House Hewett',
#                'House Hightower',
                'House Hutcheson',
                'House Inchfield',
                'House Kidwell',
                'House Leygood',
                'House Lowther',
                'House Lyberr',
                'House Meadows',
                'House Merryweather',
                'House Middlebury',
                'House Norcross',
                'House Norridge',
                'House Oakheart',
                'House Oldflowers',
                'House Orme',
                'House Peake',
                'House Rhysling',
                'House Risley',
                'House Rowan',
                'House Roxton',
                'House Serry',
                'House Shermer',
                'House Sloane',
                'House Stackhouse',
                'House Tarly',
                'House Tyrell',
                'House Pommingham',
                'House Redding',
                'House Redwyne',
                'House Uffering',
                'House Varner',
                'House Vyrwel',
                'House Westbrook',
                'House Willum',
                'House Woodwright',
                'House Wythers',
                'House Yelshire']

house_martell = ['House Allyrion',
                 'house martell',
                 'House Blackmont',
                 'House Dalt',
                 'House Dayne',
                 'House Fowler',
                 'House Gargalen',
                 'House Jordayne',
                 'House Manwoody',
                 'House Qorgyle',
                 'House Santagar',
                 'House Toland',
                 'House Uller',
                 'House Vaith',
                 'House Wyl',
                 'House Yronwood']

# creation of dictionary
dictionary_great_houses = {'House Stark':house_stark,
                           'House Lannister':house_lannister,
                           'house Arryn':house_arryn,
                           'house Tully':house_tully,
                           'house Greyjoy':house_greyjoy,
                           'house Baratheon':house_baratheon,
                           'house Tyrell':house_tyrell,
                           'house Martell':house_martell}
             

df_1 = pd.DataFrame.from_dict(dictionary_great_houses, orient='index')

great_houses = df_1.transpose()

for col in great_houses.columns:
  if great_houses[col].dtype == object:
     great_houses[col] = great_houses[col].str.lower()
     
     
#Loop to see who has an allegiance with who
 
#House Stark
got['ally_house_stark'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['House Stark']):
       got.loc[i,'ally_house_stark'] = 1
    else: got.loc[i,'ally_house_stark'] = 0

for i in range(0,len(got)):
    if got.loc[i,'culture_filled'] == 'northmen':
       got.loc[i,'ally_house_stark'] = 1


#house Lannister   
got['ally_house_lannister'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['House Lannister']):
       got.loc[i,'ally_house_lannister'] = 1
    else: got.loc[i,'ally_house_lannister'] = 0

#house Arryn   
got['ally_house_arryn'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['house Arryn']):
       got.loc[i,'ally_house_arryn'] = 1
    else: got.loc[i,'ally_house_arryn'] = 0
    
#house Tully   
got['ally_house_tully'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['house Tully']):
       got.loc[i,'ally_house_tully'] = 1
    else: got.loc[i,'ally_house_tully'] = 0


for i in range(0,len(got)):
    if got.loc[i,'culture_filled'] == 'riverlands':
       got.loc[i,'ally_house_tully'] = 1



#house Greyjoy   
got['ally_house_greyjoy'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['house Greyjoy']):
       got.loc[i,'ally_house_greyjoy'] = 1
    else: got.loc[i,'ally_house_greyjoy'] = 0
    
for i in range(0,len(got)):
    if got.loc[i,'culture_filled'] == 'ironborn':
       got.loc[i,'ally_house_greyjoy'] = 1


#house Baratheon  
got['ally_house_baratheon'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['house Baratheon']):
       got.loc[i,'ally_house_baratheon'] = 1
    else: got.loc[i,'ally_house_baratheon'] = 0   

#house Tyrell
got['ally_house_tyrell'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['house Tyrell']):
       got.loc[i,'ally_house_tyrell'] = 1
    else: got.loc[i,'ally_house_tyrell'] = 0 
 
    
    
    
#house Martell
got['ally_house_Martell'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] in list(great_houses['house Martell']):
       got.loc[i,'ally_house_Martell'] = 1
    else: got.loc[i,'ally_house_Martell'] = 0
    
#No Allies
got['no_ally'] = np.nan
    
for i in range(0,len(got)):
    if sum([got.loc[i,'ally_house_stark'],
           got.loc[i,'ally_house_lannister'],
           got.loc[i,'ally_house_arryn'],
           got.loc[i,'ally_house_tully'],
           got.loc[i,'ally_house_greyjoy'],
           got.loc[i,'ally_house_baratheon'],
           got.loc[i,'ally_house_tyrell'],
           got.loc[i,'ally_house_Martell']]) == 0:
        got.loc[i,'no_ally'] = 1
    else: got.loc[i,'no_ally'] = 0

people_w_no_allies = got[got['no_ally']==1]

people_w_no_allies.loc[:,['name','isAlive']]

got['house_filled'].value_counts()
#Multiple Allies
got['multiple_ally'] = np.nan
    
for i in range(0,len(got)):
    if sum([got.loc[i,'ally_house_stark'],
           got.loc[i,'ally_house_lannister'],
           got.loc[i,'ally_house_arryn'],
           got.loc[i,'ally_house_tully'],
           got.loc[i,'ally_house_greyjoy'],
           got.loc[i,'ally_house_baratheon'],
           got.loc[i,'ally_house_tyrell'],
           got.loc[i,'ally_house_Martell']]) >= 2:
        got.loc[i,'multiple_ally'] = 1
    else: got.loc[i,'multiple_ally'] = 0 
    
people_with_multiple_allies = got[got['multiple_ally']==1]
    

#Add nightswatch and Targaryen 

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] == 'house targaryen':
       got.loc[i,'culture_filled'] = 'valyrian'


got['nightswatch'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] == 'night\'s watch':
       got.loc[i,'nightswatch'] = 1
    else: got.loc[i,'nightswatch'] = 0

got['house frey'] = np.nan

for i in range(0,len(got)):
    if got.loc[i,'house_filled'] == 'house frey':
       got.loc[i,'house frey'] = 1
    else: got.loc[i,'house frey'] = 0



###############################################################################

# 6. Age

###############################################################################

age_lo = 0

got['out_age'] = 0


for val in enumerate(got.loc[ : , 'age']):
    
    if val[1] < age_lo:
        got.loc[val[0], 'out_age'] = 1


#Check the outliers
got[got['out_age']==1].name

#Change the values (but they are flagged)
#Rhaego
got.loc[110,'age'] = 0

#Doreah
got.loc[1350,'age'] = 25

#Flag people who are 100 years old

old_people = got[got['age']>=85]
old_people.loc[:,['name','age','isAlive']]

age_hi = 100

got['age_100'] = 0

for val in enumerate(got.loc[ : , 'age']):
    
    if val[1] >= age_hi:
        got.loc[val[0], 'age_100'] = 1

#Visualize Age
age_dropped = got.age.dropna()
sns.distplot(age_dropped)
plt.show()

###############################################################################

# 6. Date of Birth

###############################################################################

dob_dropped = got['dateOfBirth'].dropna()
dob_dropped_f = dob_dropped.drop([110,1350])
sns.distplot(dob_dropped_f)

dob_early = got[got['dateOfBirth']<=200]
dob_early.loc[:,['name','dateOfBirth','age','isAlive']]

#Flag everyone who was born before 200

dob_lo = 200

got['dob_before_200'] = 0

for val in enumerate(got.loc[ : , 'dateOfBirth']):
    
    if val[1] <= dob_lo:
        got.loc[val[0], 'dob_before_200'] = 1

###############################################################################

# 7. Books

###############################################################################

got_book_group = got.groupby(['book1_A_Game_Of_Thrones',
                              'book2_A_Clash_Of_Kings',
                              'book3_A_Storm_Of_Swords',
                              'book4_A_Feast_For_Crows',
                              'book5_A_Dance_with_Dragons'])
    
book_count = got_book_group.count()

#There are 272 characters that do not appear in any book 


missing_ppl = got[(got['book1_A_Game_Of_Thrones'] == 0) &
              (got['book2_A_Clash_Of_Kings'] == 0) &
              (got['book3_A_Storm_Of_Swords'] == 0) &
              (got['book4_A_Feast_For_Crows'] == 0) &
              (got['book5_A_Dance_with_Dragons'] == 0)].name



got['out_books'] = np.nan
list2 = list(missing_ppl)
len(list2)

for i in range(0, len(got)):
    if got.loc[i,'name'] in list2:
       got.loc[i,'out_books'] = 1
    else:
       got.loc[i,'out_books'] = 0
       
###############################################################################

# 8. Popularity and Number of Dead Relations 

###############################################################################

# single out relevant column dead relations
dead_relations = got.loc[:, 'numDeadRelations']
dead_relations.describe()

# single out relevant column popularity
popularity = got.loc[:, 'popularity']

# create a list with column headers
stats_columns = list(['min', '20%','40%','60%','80%','90%','95%', 'max'])

# create a list with min- max- and interesting quantiles
dead_relations_stats = [dead_relations.min(),
                     np.percentile(dead_relations, 20),
                     np.percentile(dead_relations, 40),
                     np.percentile(dead_relations, 60),
                     np.percentile(dead_relations, 80),
                     np.percentile(dead_relations, 90),
                     np.percentile(dead_relations, 95),
                     dead_relations.max()]

# transform list into a dataframe 
df_stats_dr = pd.DataFrame(np.array(dead_relations_stats).reshape(1,8),
                           columns = stats_columns)

print(df_stats_dr)

# create a list with min- max- and interesting quantiles 
popularity_stats = [dead_relations.min(),
                     np.percentile(popularity, 20),
                     np.percentile(popularity, 40),
                     np.percentile(popularity, 60),
                     np.percentile(popularity, 80),
                     np.percentile(popularity, 90),
                     np.percentile(popularity, 95),
                     dead_relations.max()]

# transform list into a dataframe
df_stats_pop = pd.DataFrame(np.array(popularity_stats).reshape(1,8),
                           columns = stats_columns)

print(df_stats_pop)

# initialise a new column
got['out_popular'] = 0

# mark all rows of people that are in or above the 90th percentile
for i in range(0, len(got)):
    if (got.loc[i,'popularity'] <= 0.01) :
        got.loc[i,'out_popular'] = 1
    else: got.loc[i,'out_popular'] = 0
    
# initialise a new column
got['out_dead_relation'] = 0

# NumDeadRelations
for i in range(0, len(got)):
    if got.loc[i,'numDeadRelations'] >=2:
        got.loc[i,'out_dead_relation'] = 1
    else: got.loc[i,'out_dead_relation'] = 0    
    
got['out_popular_hi'] = 0
 
for i in range(0, len(got)):
   if (got.loc[i,'popularity'] >= 0.9)  :
       got.loc[i,'out_popular_hi'] = 1
   else: got.loc[i,'out_popular_hi'] = 0
    
###############################################################################

# 9. Dummy Variables

###############################################################################     
got.loc[1:50,['title','title_filled']]
  
#Drop columns with a lot of NA's   
got_explored = got.drop(['title',
                         'culture',
                         'mother',
                         'male',
                         'father',
                         'heir',
                         'first_names',
                         'spouse',
                         'isAliveMother',
                         'isAliveFather',
                         'isAliveHeir',
                         'isAliveSpouse',
                         'out_age',
                         'house',
                         'house_filled'],
                          axis=1)


#Create dummies for title_filled_grouped
t_f_g_dummies = pd.get_dummies(list(got_explored['title_filled_grouped']),
                               drop_first = True)

#Create dummies for culture_filled
c_f_dummies = pd.get_dummies(list(got_explored['culture_filled']),
                             drop_first = True)

#Create dummies for house_complete {not in this dataset}
#h_c_dummies = pd.get_dummies(list(got_explored['house_complete']),
        #                     drop_first = True)


# Concatenating One-Hot Encoded Values with the Larger DataFrame
got_explored = pd.concat(
        [got_explored.loc[:,:],
         t_f_g_dummies, 
         c_f_dummies], 
         #h_c_dummies],
         axis = 1)


print(
      got_explored.isnull().sum()
      )


###############################################################################

# 9. Create a combination of flags

###############################################################################
#Flag combo of variables that indicate high chances of survival

got2 = got_explored.drop(['S.No','out_popular',
                                   'name',
                                   'dateOfBirth',
                                   'numDeadRelations',
                                   'age', 
                                   'popularity',
                                   'title_filled',
                                   'title_filled_grouped',
                                   'house_complete',
                                   'culture_filled'], axis = 1)

#Loop to determine what combo of variables indicate a high chance of survival
for i in range (0,81):
    for j in range (0,81):
        if i != j:
            for x in range(0,1):
                for y in range(0,1):
                    combo = got2.loc[(got2.iloc[:,i] == x) & 
                                     (got2.iloc[:,j] == y), 'isAlive']
                    if (((combo.sum()/combo.count()) > 0.85) & (len(combo)>=100)):
                        print(got2.iloc[:,[i,j]].columns,x,y)
                        
#Women in book 5
got_explored['woman_book5'] = 0

for i in range(0, len(got)):
    if (got_explored.loc[i,'book5_A_Dance_with_Dragons'] == 0 and got_explored.loc[i,'male_copy'] == 0):
        got_explored.loc[i,'woman_book5'] = 1
    else: got_explored.loc[i,'woman_book5'] = 0
    
#Not noble with a spouse
got_explored['Notnoble_spouse'] = 0

for i in range(0, len(got)):
    if (got_explored.loc[i,'isNoble'] == 0 and got_explored.loc[i,'m_spouse'] == 0):
        got_explored.loc[i,'Notnoble_spouse'] = 1
    else: got_explored.loc[i,'Notnoble_spouse'] = 0

#Female & in no allies
got_explored['female_allies'] = 0

for i in range(0, len(got)):
    if (got_explored.loc[i,'male_copy'] == 0 and got_explored.loc[i,'no_ally'] == 0):
        got_explored.loc[i,'female_allies'] = 1
    else: got_explored.loc[i,'female_allies'] = 0 
    
#Female & in books
got_explored['female_nobooks'] = 0

for i in range(0, len(got)):
    if (got_explored.loc[i,'male_copy'] == 0 and got_explored.loc[i,'out_books'] == 0):
        got_explored.loc[i,'female_nobooks'] = 1
    else: got_explored.loc[i,'female_nobooks'] = 0

#Female & Not Valyrian
got_explored['female_notvalyrian'] = 0

for i in range(0, len(got)):
    if (got_explored.loc[i,'male_copy'] == 0 and got_explored.loc[i,'valyrian'] == 0):
        got_explored.loc[i,'female_notvalyrian'] = 1
    else: got_explored.loc[i,'female_notvalyrian'] = 0
    
#House Targaryen
got_explored['house_targaryen'] = 0

for i in range(0, len(got)):
    if got_explored.loc[i,'house_complete'] == 'house targaryen':
       got_explored.loc[i,'house_targaryen'] = 1
    else: got_explored.loc[i,'house_targaryen'] = 0 
    
#House Tyrell
got_explored['house_tyrell'] = 0

for i in range(0, len(got)):
    if got_explored.loc[i,'house_complete'] == 'house tyrell':
       got_explored.loc[i,'house_tyrell'] = 1
    else: got_explored.loc[i,'house_tyrell'] = 0  

#Flag combo of variables that indicate low chances of survival                       

for i in range (0,81):
    for j in range (0,81):
        if (i != j and i != 7 and j !=7):
            for x in range(0,1):
                for y in range(0,1):
                    combo = got2.loc[(got2.iloc[:,i] == x) & 
                                     (got2.iloc[:,j] == y), 'isAlive']
                    if (((combo.sum()/combo.count()) <= 0.30) & len(combo)>=10):
                        print(got2.iloc[:,[i,j]].columns,x,y)
                        
#None

###############################################################################

# 10. Export to Excel

###############################################################################

got_explored.to_excel('got_explored5.xlsx')

