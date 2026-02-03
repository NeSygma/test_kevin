% weather.pl
raining.
is_raining :- raining.

:- initialization(main).

main :-
    ( is_raining -> writeln('Yes, it is raining.') ; writeln('No, it is not raining.') ).