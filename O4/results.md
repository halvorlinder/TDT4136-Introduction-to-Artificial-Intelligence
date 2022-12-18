## EASY
Calls (MRV): 1  
Fails (MRV): 0  
Calls (First): 1  
Fails (First): 0  
7 8 4 | 9 3 2 | 1 5 6   
6 1 9 | 4 8 5 | 3 2 7   
2 3 5 | 1 7 6 | 4 8 9   
------+-------+------  
5 7 8 | 2 6 1 | 9 3 4   
3 4 1 | 8 9 7 | 5 6 2   
9 2 6 | 5 4 3 | 8 7 1   
------+-------+------  
4 5 3 | 7 2 9 | 6 1 8   
8 6 2 | 3 1 4 | 7 9 5   
1 9 7 | 6 5 8 | 2 4 3   
  
## MEDIUM
Calls (MRV): 2  
Fails (MRV): 0  
Calls (First): 3  
Fails (First): 0  
8 7 5 | 9 3 6 | 1 4 2   
1 6 9 | 7 2 4 | 3 8 5   
2 4 3 | 8 5 1 | 6 7 9   
------+-------+------  
4 5 2 | 6 9 7 | 8 3 1   
9 8 6 | 4 1 3 | 2 5 7   
7 3 1 | 5 8 2 | 9 6 4   
------+-------+------  
5 1 7 | 3 6 9 | 4 2 8   
6 2 8 | 1 4 5 | 7 9 3   
3 9 4 | 2 7 8 | 5 1 6   
  
## HARD
Calls (MRV): 7  
Fails (MRV): 2  
Calls (First): 12  
Fails (First): 4  
1 5 2 | 3 4 6 | 8 9 7   
4 3 7 | 1 8 9 | 6 5 2   
6 8 9 | 5 7 2 | 3 1 4   
------+-------+------  
8 2 1 | 6 3 7 | 9 4 5   
5 4 3 | 8 9 1 | 7 2 6   
9 7 6 | 4 2 5 | 1 8 3   
------+-------+------  
7 9 8 | 2 5 3 | 4 6 1   
3 6 5 | 9 1 4 | 2 7 8   
2 1 4 | 7 6 8 | 5 3 9   
  
## VERYHARD
Calls (MRV): 56  
Fails (MRV): 43  
Calls (First): 68  
Fails (First): 57  
4 3 1 | 8 6 7 | 9 2 5   
6 5 2 | 4 9 1 | 3 8 7   
8 9 7 | 5 3 2 | 1 6 4   
------+-------+------  
3 8 4 | 9 7 6 | 5 1 2   
5 1 9 | 2 8 4 | 7 3 6   
2 7 6 | 3 1 5 | 8 4 9   
------+-------+------  
9 4 3 | 7 2 8 | 6 5 1   
7 6 5 | 1 4 3 | 2 9 8   
1 2 8 | 6 5 9 | 4 7 3   
  

We can see that the easier boards require less calls to backtrack, and result in fewer fails. This is slightly improved using MRV, as the variable with the smallest domain is selected in each iteration, making the search smaller. I would think that the easier problems are easier because the domains of the variables become small after applying ID3, resulting in few calls to backtrack.