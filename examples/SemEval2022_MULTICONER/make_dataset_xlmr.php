<?php

$dir="data_xlmr";

@mkdir($dir);

$sid=0;

$tags=[];

function startsWith( $haystack, $needle ) {
     $length = strlen( $needle );
     return substr( $haystack, 0, $length ) === $needle;
}
function endsWith( $haystack, $needle ) {
    $length = strlen( $needle );
    if( !$length ) {
        return true;
    }
    return substr( $haystack, -$length ) === $needle;
}

function processFile($fnameIn, $fnameOut){
    global $sid,$tags;
    $fout=fopen($fnameOut,"w");
    $lastEmpty=true;
    foreach(explode("\n",file_get_contents($fnameIn)) as $line){
        if(startsWith($line,"# id"))continue;
        $ldata=explode(" ",$line);
        if(count($ldata)!=4 && count($ldata)!=3){
            fwrite($fout,"\n");
            $lastEmpty=true;
        }else{
            if(count($ldata)==3)$ldata[3]="O";
            //if($lastEmpty===true){fwrite($fout,"# id = ${sid}\n");$sid++;}
            fwrite($fout,"${ldata[0]} _ _ ${ldata[3]}\n");
            $lastEmpty=false;
            if(!isset($tags[$ldata[3]]))$tags[$ldata[3]]=count($tags);
        }
    }
    if(!$lastEmpty)fwrite($fout,"\n");
    fclose($fout);
}

function processLang($lshort,$lang){
    global $dir;
    @mkdir("$dir/$lshort");
    processFile("./data_test/$lang/${lshort}_dev.conll","$dir/$lshort/valid.txt");
    processFile("./data_test/$lang/${lshort}_test.conll","$dir/$lshort/test.txt");
    processFile("./data_test/$lang/${lshort}_train.conll","$dir/$lshort/train.txt");
}

processLang("en","EN-English");
processLang("bn","BN-Bangla");
processLang("de","DE-German");
processLang("es","ES-Spanish");
processLang("fa","FA-Farsi");
processLang("hi","HI-Hindi");
processLang("ko","KO-Korean");
processLang("nl","NL-Dutch");
processLang("ru","RU-Russian");
processLang("tr","TR-Turkish");
processLang("zh","ZH-Chinese");
processLang("mix","MIX_Code_mixed");
processLang("multi","MULTI_Multilingual");

//var_dump($tags);
foreach($tags as $t=>$i){
    echo "'$t': $i,";
}

echo "\n\n";

foreach($tags as $t=>$i){
    echo "$t,";
}
echo "\n";


