<?php

$dir="data_xlmr_3";

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
        if(startsWith($line,"# "))continue;
        $ldata=explode("\t",$line);
        if(count($ldata)<8){
            fwrite($fout,"\n");
            $lastEmpty=true;
        }else{
            $w=$ldata[1];
            $tag=$ldata[count($ldata)-1];
            if($tag=="_")$tag="O";
            fwrite($fout,"${w} _ _ ${tag}\n");
            $lastEmpty=false;
            if(!isset($tags[$tag]))$tags[$tag]=count($tags);
        }
    }
    if(!$lastEmpty)fwrite($fout,"\n");
    fclose($fout);
}

@mkdir("./data_xlmr_3");
processFile("./data_conllup_3/bio_simo_dev.conllup","./data_xlmr_3/valid.txt");
processFile("./data_conllup_3/bio_simo_test.conllup","./data_xlmr_3/test.txt");
processFile("./data_conllup_3/bio_simo_train.conllup","./data_xlmr_3/train.txt");

//var_dump($tags);
foreach($tags as $t=>$i){
    echo "'$t': $i,";
}

echo "\n\n";

foreach($tags as $t=>$i){
    echo "$t,";
}
echo "\n";


