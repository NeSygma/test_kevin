import clingo

asp_program = """
{a; b}.
:- a, b.
#show a/0.
#show b/0.
"""

ctl = clingo.Control(["0"])
ctl.add("base", [], asp_program)
ctl.ground([("base", [])])

# simple counter in a mutable container so callback can update it
count = {"n": 0}

def on_model(model):
    count["n"] += 1
    # get shown symbols and convert to readable strings
    shown = model.symbols(shown=True)
    shown_strs = [str(s) for s in shown]
    print(f"Answer set {count['n']}: {shown_strs}")

ctl.solve(on_model=on_model)
