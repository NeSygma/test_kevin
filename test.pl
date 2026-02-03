% file: family.pl

% === Facts ===
father(john, mary).
father(john, david).
mother(susan, mary).
mother(susan, david).

% === Rules ===
parent(X, Y) :- father(X, Y).
parent(X, Y) :- mother(X, Y).

sibling(X, Y) :-
    father(F, X), father(F, Y),
    mother(M, X), mother(M, Y),
    X \= Y.

% === Main Entry Point ===
main :-
    write('Parents of mary:'), nl,
    parent(X, mary),
    format('- ~w~n', [X]),
    fail.  % force backtracking to get all solutions

main :- 
    nl, write('Siblings of david:'), nl,
    sibling(david, Sibling),
    format('- ~w~n', [Sibling]),
    fail.

main.  % prevent failure after all results are shown

% === Initialization Directive ===
:- initialization(main).


