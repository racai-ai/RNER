<?php

$entities=[];

function make_entity_list(){

    global $entities;

    $current=[];
    $currentType="";

    $fin=fopen("data_xlmr/multi/train.txt","r");
    while(!feof($fin)){
        $line=fgets($fin);
        if($line===false)break;
        $line=rtrim($line);
        $ldata=explode(" ",$line);

        $w="";
        $n="O";

        if(count($ldata)==4){

            $w=$ldata[0];
            $n=$ldata[3];
        }

        if($n=="O"){
            if($currentType!=""){
                if(!isset($entities[$currentType]))$entities[$currentType]=[];
                $entities[$currentType][]=$current;

                $currentType="";
                $current=[];
            }

            continue;
        }

        $type=substr($n,2);
        if($n[0]=="B"){

            if($currentType!=""){
                if(!isset($entities[$currentType]))$entities[$currentType]=[];
                $entities[$currentType][]=$current;
            }

            $currentType=$type;
            $current=[];
            $current[]=$w;
        }else{
            $current[]=$w;
        }

    }

    fclose($fin);

}

function make_line($ldata){return implode(" ",$ldata);}

function augment_sent($sent){
    global $entities;

    $ret=[];
    for($i=0;$i<count($sent);$i++){
        if($sent[$i][3]=="O")$ret[]=$sent[$i];
        else if($sent[$i][3][0]=="B"){
            $type=substr($sent[$i][3],2);
            $r=rand(0,count($entities[$type])-1);
            $first=true;
            foreach($entities[$type][$r] as $w){
                $t="I-";
                if($first){$t="B-";$first=false;}
                $ret[]=[$w,"_","_",$t.$type];
            }
        }// else {;} // ignore I-
    }
    return $ret;
}

function augment(){
    $fin=fopen("./data_xlmr/mix/train.txt","r");
    $fout=fopen("./data_xlmr/mix/train_augmented.txt","w");

    $sent=[];

    while(!feof($fin)){
        $line=fgets($fin);
        if($line===false)break;

        $line=rtrim($line);
        $ldata=explode(" ",$line);
        if(count($ldata)!=4){
            if(count($sent)==0)continue;
            $sent2=augment_sent($sent);
            fwrite($fout,implode("\n",array_map('make_line',$sent)));
            fwrite($fout,"\n\n");
            fwrite($fout,implode("\n",array_map('make_line',$sent2)));
            fwrite($fout,"\n\n");
            $sent=[];
            continue;
        }

        $sent[]=$ldata;


    }

    fclose($fin);
    fclose($fout);
}

echo "Preparing entity list ... ";
make_entity_list();
echo "Done\nAugmenting ... ";
augment();
echo "Done\n";

