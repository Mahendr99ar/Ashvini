# Problem brief

## The problem

Low Earth orbit is getting crowded. Operators receive conjunction warnings whenever another object is predicted to pass close to one of their satellites. Most warnings never turn into real danger, but a few do, and the operator has to decide which few, usually with days or hours to spare.

Each wrong answer costs something:

- **Manoeuvre when it wasn't needed:** propellant spent, and propellant is the satellite's remaining life. The satellite may also miss imaging or service time while it moves.
- **No manoeuvre when it was needed:** a collision, the loss of the satellite, and a debris cloud that threatens everyone else at that altitude for years.

ISRO alone carried out 122 collision avoidance manoeuvres in the 14 years covered by its ISSAR 2024 report, and only a small share of the warnings an operator receives ever end in a manoeuvre.

## Who has it

- Satellite operators, from agencies to small startups with one or two satellites and no dedicated flight-dynamics team.
- Space-safety analysts who watch the whole environment, not one fleet.
- Researchers, journalists and policy people who need a trustworthy picture of what is happening up there.

## What people do today

- Large operators run their own flight-dynamics tools and buy commercial tracking data.
- Small operators often rely on the free warnings from Space-Track and a spreadsheet.
- Commercial services (LeoLabs, Slingshot, Digantara and others) sell tracking and screening.

## The gap Ashvini goes after

1. **Triage.** Turning a stream of warnings into a short list, ranked by how likely each one is to end up dangerous, with an honest uncertainty.
2. **Context.** Showing space weather, orbit uncertainty and the history of each event in one place, so the decision does not depend on five tabs.
3. **Small operators.** Something a two-person team can run for free, on public data, that speaks their file formats.

## What Ashvini is not

- Not an official warning service. Public orbital data is less precise than what operators and the Space Force hold.
- Not an autonomous system. It recommends; a person decides.

## How we will know it works

See `metrics.md`. In one line: an operator using Ashvini should be able to dismiss more false alarms without missing any event that turns out to be high risk.
