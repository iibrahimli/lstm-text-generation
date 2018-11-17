# lstm-text-generation
An LSTM written using Keras to generate text character by character
It was trained on C code I wrote for my Data Structures and Algorithms class, which was around 1000 lines maximum.
It overfit on that data, as it is seen from nearly intelligible variable names and string literals.

For example:
`int ll_add(snode **pplist,int n){
  snode *ptemt, *pcur;
  psent=pcur=*pplist;
  while((pcur)&&(ptrcmp((pcur->wsword).s,s))){
    psent=pcur;
    pcur=pcur->pnext;
  }
  if (pcur==null) break; 
  }
  if (pcur){
    if (pcur==*pplist) *pplist=pcur->pnext;
    else psent->pnext=pcur->pnext;
    free(pcur);
    return 0;
  } else return 1;
}`

This is from the output of 2-layer LSTM, which is larger than the network used to generate `c_generated_example.txt`, which gave it more expressive power and allowed it to easily overfit.
It has learned what a function generally looks like return type, name, braces, parentheses, etc.
Variable names are more or less the same as in my original code.


Examples can be found in files:
 1) `c_generated_example.txt`   --  few epochs of training, basically outputs garbage
 2) `c_generated_2_layers_50_epochs.txt`  --  a 2 layer LSTM trained for 50 epochs which allowed it to nicely overfit the data
