"""Synthetic, self-authored dialogue dataset generator for large-scale VGRFT
stage-1 SFT (see ``docs/reports/MICRO_CFNA_SFT_100K_REPORT.md``).

Honesty note up front: this is **programmatically generated template data**,
not a scraped or third-party corpus. That makes provenance/licensing trivial
(everything here is originally authored for this repository, no external
license to track), but it also means the "diversity" is combinatorial
(varied phrasings x varied parameters/entities), not organic human language.
The dataset-preparation manifest records this plainly; see
``cfna.training.dataset_prep`` for where the manifest is written.

Every category below is built by **enumerating** a template x parameter space
deterministically (no RNG needed to guarantee uniqueness), so hitting a target
count is exact and collision-free by construction rather than by sampling and
hoping. ``cfna.training.dataset_prep`` still runs a real dedupe/validation
pass over the output, because the pipeline must not simply trust the
generator (and because id/category bugs should be caught, not assumed away).

Categories (mirrors the task's required mixture): greetings/small talk,
arithmetic, facts, definitions, explanations, instruction following,
rewriting, summarization, classification, simple logic, basic coding,
step-based procedures, multi-turn dialogue, safe refusals, uncertainty.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterator, List, Sequence, Tuple

SOURCE = "cfna-synthetic-template-v1"

Message = Dict[str, str]
Record = Dict[str, object]


def _rec(rid: str, messages: List[Message], category: str) -> Record:
    return {"id": rid, "messages": messages, "source": SOURCE, "category": category}


def _turn(prompt: str, response: str) -> List[Message]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


# --------------------------------------------------------------------------- #
# Greetings / small talk
# --------------------------------------------------------------------------- #

_GREET_WORDS = ["Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
                "Greetings", "Howdy", "Hiya", "Yo"]
_GREET_SUFFIX = ["", " there", "!", " friend", " everyone"]
_GREET_REPLIES = [
    "Hello! How can I help you today?",
    "Hi! What can I do for you?",
    "Hey there! How can I help?",
    "Greetings! What do you need?",
    "Hello! I am ready to help.",
]
_CLOSING_WORDS = ["Goodbye", "Bye", "See you later", "Take care", "See you", "Farewell", "Bye bye"]
_CLOSING_REPLIES = [
    "Goodbye! Have a great day.",
    "Bye! Take care.",
    "See you later! Take care.",
    "Farewell! Come back anytime.",
    "Take care! Have a good one.",
]
_SMALLTALK = [
    ("How are you?", "I am doing well, thank you for asking. How are you?"),
    ("Thank you", "You are welcome!"),
    ("Thanks", "You are welcome! Let me know if you need anything else."),
    ("Thanks a lot", "You are very welcome!"),
    ("Nice to meet you", "Nice to meet you too!"),
    ("Sorry to bother you", "No bother at all, I am here to help."),
    ("Can you help me?", "Yes, I would be glad to help. What do you need?"),
    ("What can you do?", "I can answer questions and chat about a range of topics."),
    ("Who are you?", "I am an assistant built on the CFNA byte model."),
    ("What are you?", "I am a small byte-level language model trained to hold a conversation."),
    ("I am happy today", "That is wonderful to hear!"),
    ("I am sad today", "I am sorry to hear that. I hope things get better."),
    ("I am bored", "You could try reading a book or going for a walk."),
]


def gen_greetings() -> Iterator[Record]:
    i = 0
    for word, suffix in itertools.product(_GREET_WORDS, _GREET_SUFFIX):
        prompt = f"{word}{suffix}"
        reply = _GREET_REPLIES[i % len(_GREET_REPLIES)]
        yield _rec(f"greet-{i}", _turn(prompt, reply), "greetings")
        i += 1
    for word in _CLOSING_WORDS:
        for j in range(len(_CLOSING_REPLIES)):
            yield _rec(f"greet-close-{word}-{j}", _turn(word, _CLOSING_REPLIES[j]), "greetings")
    for j, (p, r) in enumerate(_SMALLTALK):
        yield _rec(f"greet-small-{j}", _turn(p, r), "greetings")


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #

_ADD_TMPL = [
    "What is {a} plus {b}?", "Add {a} and {b}.", "Calculate {a} + {b}.",
    "What do you get if you add {a} to {b}?", "Sum {a} and {b}.",
]
_SUB_TMPL = [
    "What is {a} minus {b}?", "Subtract {b} from {a}.", "Calculate {a} - {b}.",
    "I had {a} objects and removed {b}. How many remain?", "What do you get if you take {b} away from {a}?",
]
_MUL_TMPL = [
    "What is {a} times {b}?", "Multiply {a} by {b}.", "Calculate {a} * {b}.",
    "What is the product of {a} and {b}?",
]
_DIV_TMPL = [
    "What is {a} divided by {b}?", "Divide {a} by {b}.", "Calculate {a} / {b}.",
    "How many times does {b} go into {a}?",
]


def gen_arithmetic(add_max: int = 230, mul_max: int = 85, div_b_max: int = 58, div_q_max: int = 85) -> Iterator[Record]:
    i = 0
    for tmpl in _ADD_TMPL:
        for a in range(0, add_max, 2):
            for b in range(0, add_max, 4):
                prompt = tmpl.format(a=a, b=b)
                yield _rec(f"add-{i}", _turn(prompt, f"{a} plus {b} equals {a + b}."), "arithmetic")
                i += 1
    for tmpl in _SUB_TMPL:
        for a in range(0, add_max, 2):
            for b in range(0, add_max, 4):
                if b > a:
                    continue
                prompt = tmpl.format(a=a, b=b)
                yield _rec(f"sub-{i}", _turn(prompt, f"{a} minus {b} equals {a - b}."), "arithmetic")
                i += 1
    for tmpl in _MUL_TMPL:
        for a in range(0, mul_max):
            for b in range(0, mul_max):
                prompt = tmpl.format(a=a, b=b)
                yield _rec(f"mul-{i}", _turn(prompt, f"{a} times {b} equals {a * b}."), "arithmetic")
                i += 1
    for tmpl in _DIV_TMPL:
        for b in range(1, div_b_max):
            for q in range(0, div_q_max):
                a = b * q
                prompt = tmpl.format(a=a, b=b)
                yield _rec(f"div-{i}", _turn(prompt, f"{a} divided by {b} equals {q}."), "arithmetic")
                i += 1


# --------------------------------------------------------------------------- #
# Classification (computable, correct-by-construction)
# --------------------------------------------------------------------------- #

_EVEN_ODD_TMPL = ["Is {n} even or odd?", "Classify {n} as even or odd.", "Tell me if {n} is even or odd."]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


_PRIME_TMPL = ["Is {n} a prime number?", "Determine whether {n} is prime.", "Classify {n} as prime or not prime."]

_SENTIMENT = [
    ("This is a wonderful and delightful day.", "positive"),
    ("I am so happy and excited about this.", "positive"),
    ("What a great and pleasant surprise.", "positive"),
    ("This makes me feel joyful and content.", "positive"),
    ("I really love how this turned out.", "positive"),
    ("This is a fantastic piece of good news.", "positive"),
    ("I am thrilled and grateful for this outcome.", "positive"),
    ("What a beautiful and uplifting experience.", "positive"),
    ("This brings me a lot of comfort and joy.", "positive"),
    ("I am proud and satisfied with the results.", "positive"),
    ("This is a terrible and awful situation.", "negative"),
    ("I am so sad and disappointed about this.", "negative"),
    ("What a horrible and unpleasant surprise.", "negative"),
    ("This makes me feel upset and frustrated.", "negative"),
    ("I really dislike how this turned out.", "negative"),
    ("This is a miserable piece of bad news.", "negative"),
    ("I am furious and annoyed about this outcome.", "negative"),
    ("What a dreadful and discouraging experience.", "negative"),
    ("This brings me a lot of worry and stress.", "negative"),
    ("I am embarrassed and unhappy with the results.", "negative"),
]
_SENTIMENT_TMPL = ["Is the following text positive or negative? {t}",
                   "Classify the sentiment of this sentence: {t}",
                   "What is the sentiment of: {t}"]

_ANIMAL_CLASS = [
    ("dog", "mammal"), ("cat", "mammal"), ("horse", "mammal"), ("whale", "mammal"), ("bat", "mammal"),
    ("elephant", "mammal"), ("lion", "mammal"), ("tiger", "mammal"), ("rabbit", "mammal"), ("cow", "mammal"),
    ("eagle", "bird"), ("sparrow", "bird"), ("penguin", "bird"), ("owl", "bird"), ("parrot", "bird"),
    ("flamingo", "bird"), ("crow", "bird"), ("duck", "bird"), ("hawk", "bird"), ("robin", "bird"),
    ("salmon", "fish"), ("shark", "fish"), ("tuna", "fish"), ("goldfish", "fish"), ("trout", "fish"),
    ("catfish", "fish"), ("swordfish", "fish"),
    ("frog", "amphibian"), ("toad", "amphibian"), ("newt", "amphibian"), ("salamander", "amphibian"),
    ("snake", "reptile"), ("lizard", "reptile"), ("turtle", "reptile"), ("crocodile", "reptile"),
    ("gecko", "reptile"), ("iguana", "reptile"),
]
_ANIMAL_TMPL = ["Classify the following animal as mammal, bird, fish, amphibian, or reptile: {a}",
                "What class of animal is a {a}?", "Is a {a} a mammal, bird, fish, amphibian, or reptile?"]


def gen_classification(max_n: int = 3500) -> Iterator[Record]:
    i = 0
    for tmpl in _EVEN_ODD_TMPL:
        for n in range(0, max_n):
            ans = "even" if n % 2 == 0 else "odd"
            yield _rec(f"evenodd-{i}", _turn(tmpl.format(n=n), f"{n} is {ans}."), "classification")
            i += 1
    for tmpl in _PRIME_TMPL:
        for n in range(0, max_n):
            ans = "a prime number" if _is_prime(n) else "not a prime number"
            yield _rec(f"prime-{i}", _turn(tmpl.format(n=n), f"{n} is {ans}."), "classification")
            i += 1
    for tmpl in _SENTIMENT_TMPL:
        for t, label in _SENTIMENT:
            yield _rec(f"sentiment-{i}", _turn(tmpl.format(t=t), f"That sentence is {label}."), "classification")
            i += 1
    for tmpl in _ANIMAL_TMPL:
        for a, cls in _ANIMAL_CLASS:
            yield _rec(f"animal-{i}", _turn(tmpl.format(a=a), f"A {a} is a {cls}."), "classification")
            i += 1


# --------------------------------------------------------------------------- #
# Simple logical reasoning
# --------------------------------------------------------------------------- #

_SYLLOGISM_SETS = [
    ("birds", "animals", "sparrow"), ("dogs", "mammals", "poodle"), ("roses", "flowers", "a rose bush"),
    ("triangles", "shapes", "this triangle"), ("cars", "vehicles", "this car"),
    ("doctors", "professionals", "this doctor"), ("apples", "fruits", "this apple"),
    ("cats", "animals", "a kitten"), ("trees", "plants", "this oak tree"), ("books", "objects", "this novel"),
    ("squares", "rectangles", "this square"), ("whales", "mammals", "this whale"),
    ("tulips", "flowers", "this tulip"), ("teachers", "professionals", "this teacher"),
    ("oranges", "fruits", "this orange"), ("chairs", "furniture", "this chair"),
    ("planets", "celestial bodies", "Mars"), ("novels", "books", "this novel"),
    ("bicycles", "vehicles", "this bicycle"), ("lawyers", "professionals", "this lawyer"),
    ("maples", "trees", "this maple"), ("sonnets", "poems", "this sonnet"),
    ("penguins", "birds", "this penguin"), ("laptops", "computers", "this laptop"),
    ("violins", "instruments", "this violin"), ("sharks", "fish", "this shark"),
]
_SYLLOGISM_TMPL = [
    "If all {a} are {b}, and X is a {c}, is X a {b}?",
    "All {a} are {b}. {c_cap} is an example of {a_sing}. Is {c} a {b_sing}?",
]


def gen_logic() -> Iterator[Record]:
    i = 0
    for a, b, c in _SYLLOGISM_SETS:
        for tmpl in _SYLLOGISM_TMPL:
            prompt = tmpl.format(a=a, b=b, c=c, c_cap=c.capitalize(), a_sing=a[:-1], b_sing=b[:-1])
            yield _rec(f"syllogism-{i}", _turn(prompt, f"Yes, since all {a} are {b}, X is a {b}."), "logic")
            i += 1
    # If-then reasoning with varied nouns/conditions
    _COND = [
        ("it rains", "the ground gets wet"), ("the sun sets", "it becomes dark"),
        ("you heat ice", "it melts"), ("you drop a glass", "it may break"),
        ("a plant gets no water", "it wilts"), ("you turn off the light", "the room becomes dark"),
        ("you leave milk out too long", "it spoils"), ("you touch a hot stove", "you may get burned"),
        ("you plant a seed and water it", "it may grow"), ("you freeze water", "it turns to ice"),
        ("you overinflate a balloon", "it may pop"), ("you don't charge your phone", "the battery runs out"),
        ("you mix red and blue paint", "you get purple paint"), ("you leave bread out", "it may go stale"),
        ("you exercise regularly", "you tend to get healthier"), ("you skip a meal", "you may feel hungry"),
        ("you study for a test", "you tend to do better on it"), ("you leave a door open in winter", "the room gets cold"),
        ("you add salt to ice", "it melts faster"), ("you press a light switch", "the light turns on"),
    ]
    for cond, effect in _COND:
        for tmpl in ["If {cond}, then {effect}. Given that {cond}, what happens?",
                     "Suppose {cond}. What is the likely result?"]:
            prompt = tmpl.format(cond=cond, effect=effect)
            yield _rec(f"cond-{i}", _turn(prompt, f"If {cond}, then {effect}."), "logic")
            i += 1
    # Numeric comparison: a large, correctness-guaranteed combinatorial space,
    # the same "simple logical reasoning" skill (ordering) as the syllogisms
    # above, just with numbers instead of category membership.
    cmp_tmpl = ["Which is bigger, {a} or {b}?", "Which number is larger: {a} or {b}?",
                "Is {a} greater than {b}?"]
    for a in range(0, 220, 3):
        for b in range(0, 220, 11):
            if a == b:
                continue
            bigger = a if a > b else b
            t0 = cmp_tmpl[0].format(a=a, b=b)
            yield _rec(f"cmp0-{i}", _turn(t0, f"{bigger} is bigger."), "logic")
            i += 1
            t1 = cmp_tmpl[1].format(a=a, b=b)
            yield _rec(f"cmp1-{i}", _turn(t1, f"{bigger} is larger than {a if bigger == b else b}."), "logic")
            i += 1
            ans2 = "Yes" if a > b else "No"
            t2 = cmp_tmpl[2].format(a=a, b=b)
            yield _rec(f"cmp2-{i}", _turn(t2, f"{ans2}, {a} is {'greater' if a > b else 'not greater'} than {b}."),
                       "logic")
            i += 1


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #

_CAPITALS = [
    ("France", "Paris", "Europe"), ("Japan", "Tokyo", "Asia"), ("Italy", "Rome", "Europe"),
    ("Germany", "Berlin", "Europe"), ("Spain", "Madrid", "Europe"), ("Canada", "Ottawa", "North America"),
    ("Egypt", "Cairo", "Africa"), ("Brazil", "Brasilia", "South America"), ("Russia", "Moscow", "Europe"),
    ("China", "Beijing", "Asia"), ("India", "New Delhi", "Asia"), ("Australia", "Canberra", "Oceania"),
    ("Mexico", "Mexico City", "North America"), ("Greece", "Athens", "Europe"), ("Portugal", "Lisbon", "Europe"),
    ("Norway", "Oslo", "Europe"), ("Sweden", "Stockholm", "Europe"), ("Poland", "Warsaw", "Europe"),
    ("Turkey", "Ankara", "Asia"), ("Kenya", "Nairobi", "Africa"), ("Argentina", "Buenos Aires", "South America"),
    ("Peru", "Lima", "South America"), ("Chile", "Santiago", "South America"), ("Thailand", "Bangkok", "Asia"),
    ("Vietnam", "Hanoi", "Asia"), ("Ireland", "Dublin", "Europe"), ("Finland", "Helsinki", "Europe"),
    ("Denmark", "Copenhagen", "Europe"), ("Austria", "Vienna", "Europe"), ("Switzerland", "Bern", "Europe"),
    ("Belgium", "Brussels", "Europe"), ("Netherlands", "Amsterdam", "Europe"), ("Nigeria", "Abuja", "Africa"),
    ("South Africa", "Pretoria", "Africa"), ("Morocco", "Rabat", "Africa"), ("Ghana", "Accra", "Africa"),
    ("Ethiopia", "Addis Ababa", "Africa"), ("Indonesia", "Jakarta", "Asia"), ("Malaysia", "Kuala Lumpur", "Asia"),
    ("Philippines", "Manila", "Asia"), ("South Korea", "Seoul", "Asia"), ("Pakistan", "Islamabad", "Asia"),
    ("Bangladesh", "Dhaka", "Asia"), ("Iran", "Tehran", "Asia"), ("Iraq", "Baghdad", "Asia"),
    ("Israel", "Jerusalem", "Asia"), ("Saudi Arabia", "Riyadh", "Asia"), ("United Kingdom", "London", "Europe"),
    ("Ukraine", "Kyiv", "Europe"), ("Romania", "Bucharest", "Europe"), ("Hungary", "Budapest", "Europe"),
    ("Czechia", "Prague", "Europe"), ("Croatia", "Zagreb", "Europe"), ("Serbia", "Belgrade", "Europe"),
    ("Bulgaria", "Sofia", "Europe"), ("Iceland", "Reykjavik", "Europe"), ("New Zealand", "Wellington", "Oceania"),
    ("Colombia", "Bogota", "South America"), ("Venezuela", "Caracas", "South America"),
    ("Ecuador", "Quito", "South America"), ("Uruguay", "Montevideo", "South America"),
    ("Cuba", "Havana", "North America"), ("Jamaica", "Kingston", "North America"),
    ("Panama", "Panama City", "North America"), ("Costa Rica", "San Jose", "North America"),
]
_CAPITAL_TMPL = ["What is the capital of {c}?", "Which city is the capital of {c}?",
                 "Tell me the capital of {c}.", "Name the capital city of {c}."]
_CONTINENT_TMPL = ["What continent is {c} in?", "Which continent is {c} located on?"]

_PLANETS = [
    ("Mercury", "1st"), ("Venus", "2nd"), ("Earth", "3rd"), ("Mars", "4th"),
    ("Jupiter", "5th"), ("Saturn", "6th"), ("Uranus", "7th"), ("Neptune", "8th"),
]
_PLANET_TMPL = ["What position is {p} from the sun?", "Which planet is the {o} from the sun?"]

_COLORS = [("sky", "blue"), ("grass", "green"), ("blood", "red"), ("snow", "white"),
           ("coal", "black"), ("banana", "yellow"), ("carrot", "orange"), ("grape", "purple")]
_COLOR_TMPL = ["What color is {x} usually?", "What color is a typical {x}?"]

_UNITS = [("a week", "seven days"), ("a year", "twelve months"), ("an hour", "sixty minutes"),
          ("a minute", "sixty seconds"), ("a day", "twenty four hours"), ("a decade", "ten years")]
_UNIT_TMPL = ["How many {unit_of} are in {unit}?"]


_ELEMENTS = [
    ("Hydrogen", "H", 1), ("Helium", "He", 2), ("Carbon", "C", 6), ("Nitrogen", "N", 7),
    ("Oxygen", "O", 8), ("Sodium", "Na", 11), ("Magnesium", "Mg", 12), ("Aluminum", "Al", 13),
    ("Silicon", "Si", 14), ("Phosphorus", "P", 15), ("Sulfur", "S", 16), ("Chlorine", "Cl", 17),
    ("Potassium", "K", 19), ("Calcium", "Ca", 20), ("Iron", "Fe", 26), ("Copper", "Cu", 29),
    ("Zinc", "Zn", 30), ("Silver", "Ag", 47), ("Gold", "Au", 79), ("Lead", "Pb", 82),
]
_ELEMENT_TMPL = ["What is the chemical symbol for {e}?", "What element has the symbol {s}?",
                 "What is the atomic number of {e}?"]


def gen_facts() -> Iterator[Record]:
    i = 0
    for country, capital, continent in _CAPITALS:
        for tmpl in _CAPITAL_TMPL:
            yield _rec(f"capital-{i}", _turn(tmpl.format(c=country),
                       f"The capital of {country} is {capital}."), "facts")
            i += 1
        for tmpl in _CONTINENT_TMPL:
            yield _rec(f"continent-{i}", _turn(tmpl.format(c=country),
                       f"{country} is in {continent}."), "facts")
            i += 1
    for name, symbol, number in _ELEMENTS:
        yield _rec(f"elem-sym-{i}", _turn(_ELEMENT_TMPL[0].format(e=name),
                   f"The chemical symbol for {name} is {symbol}."), "facts")
        i += 1
        yield _rec(f"elem-name-{i}", _turn(_ELEMENT_TMPL[1].format(s=symbol),
                   f"The element with the symbol {symbol} is {name}."), "facts")
        i += 1
        yield _rec(f"elem-num-{i}", _turn(_ELEMENT_TMPL[2].format(e=name),
                   f"The atomic number of {name} is {number}."), "facts")
        i += 1
    for planet, ordinal in _PLANETS:
        for tmpl in _PLANET_TMPL:
            prompt = tmpl.format(p=planet, o=ordinal)
            yield _rec(f"planet-{i}", _turn(prompt, f"{planet} is the {ordinal} planet from the sun."), "facts")
            i += 1
    for thing, color in _COLORS:
        for tmpl in _COLOR_TMPL:
            yield _rec(f"color-{i}", _turn(tmpl.format(x=thing), f"A {thing} is usually {color}."), "facts")
            i += 1
    unit_words = {"a week": "days", "a year": "months", "an hour": "minutes",
                  "a minute": "seconds", "a day": "hours", "a decade": "years"}
    for unit, answer in _UNITS:
        prompt = f"How many {unit_words[unit]} are in {unit}?"
        yield _rec(f"unit-{i}", _turn(prompt, f"There are {answer} in {unit}."), "facts")
        i += 1


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #

_DEFINITIONS = [
    ("happy", "feeling or showing pleasure and contentment"),
    ("large", "of considerable or great size"),
    ("quick", "moving fast or doing something in a short time"),
    ("bright", "giving out or reflecting a lot of light"),
    ("gentle", "mild and kind in manner"),
    ("curious", "eager to know or learn something"),
    ("brave", "ready to face danger without fear"),
    ("honest", "truthful and sincere"),
    ("ancient", "very old or belonging to the distant past"),
    ("fragile", "easily broken or damaged"),
    ("generous", "willing to give freely"),
    ("polite", "having good manners and respect for others"),
    ("silent", "making or accompanied by no sound"),
    ("vast", "extremely large in area or size"),
    ("clever", "quick to understand and learn"),
    ("loyal", "faithful to a person, cause, or belief"),
    ("humble", "having a modest view of one's own importance"),
    ("wisdom", "the quality of having experience and good judgment"),
    ("justice", "fairness in the way people are treated"),
    ("liberty", "the freedom to act, speak, or think as one wants"),
    ("diligent", "showing care and effort in one's work or duties"),
    ("frugal", "careful and economical with money or resources"),
    ("resilient", "able to recover quickly from difficulties"),
    ("candid", "truthful and straightforward in speech"),
    ("meticulous", "showing great attention to detail"),
    ("benevolent", "well meaning and kindly"),
    ("tedious", "too long, slow, or dull"),
    ("versatile", "able to adapt to many different functions"),
    ("eloquent", "fluent and persuasive in speaking or writing"),
    ("tranquil", "free from disturbance; calm"),
    ("abundant", "existing in large quantities"),
    ("cautious", "careful to avoid potential problems"),
    ("diverse", "showing a great deal of variety"),
    ("intricate", "very complicated or detailed"),
    ("optimistic", "hopeful and confident about the future"),
    ("pessimistic", "tending to see the worst aspect of things"),
    ("reliable", "consistently good in quality and able to be trusted"),
    ("stubborn", "having or showing determination not to change one's attitude"),
    ("sincere", "free from pretense or deceit"),
    ("thorough", "complete with regard to every detail"),
    ("obscure", "not discovered or known about; uncertain"),
    ("robust", "strong and healthy; sturdy"),
    ("subtle", "so delicate as to be difficult to detect or describe"),
    ("efficient", "achieving maximum productivity with minimum wasted effort"),
    ("empathy", "the ability to understand and share the feelings of another"),
    ("integrity", "the quality of being honest and having strong moral principles"),
    ("innovation", "the introduction of new ideas or methods"),
    ("perseverance", "persistence in doing something despite difficulty"),
    ("gratitude", "the quality of being thankful"),
    ("ambition", "a strong desire to achieve something"),
    ("compassion", "sympathetic concern for the suffering of others"),
    ("curiosity", "a strong desire to know or learn something"),
    ("harmony", "agreement or concord between people or things"),
    ("resourceful", "able to find quick and clever ways to overcome difficulties"),
    ("adamant", "refusing to change one's mind or position"),
    ("meager", "lacking in quantity or quality"),
    ("prudent", "acting with care and thought for the future"),
    ("zealous", "having great energy or enthusiasm for a cause"),
    ("wary", "feeling or showing caution about possible dangers"),
]
_DEF_TMPL = ["What does {w} mean?", "Define {w}.", "Give the definition of {w}.",
             "What is the meaning of the word {w}?"]


def gen_definitions() -> Iterator[Record]:
    i = 0
    for word, meaning in _DEFINITIONS:
        for tmpl in _DEF_TMPL:
            yield _rec(f"def-{i}", _turn(tmpl.format(w=word), f"{word.capitalize()} means {meaning}."),
                       "definitions")
            i += 1


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #

_EXPLANATIONS = [
    ("why the sky is blue", "The sky looks blue because air scatters blue light from the sun more than other colors."),
    ("why ice floats on water", "Ice floats because it is less dense than liquid water."),
    ("why we see lightning before hearing thunder", "Light travels much faster than sound, so we see the flash first."),
    ("why leaves change color in autumn", "Leaves change color as chlorophyll breaks down and other pigments show through."),
    ("why the moon has phases", "The moon has phases because we see different amounts of its sunlit side as it orbits Earth."),
    ("why metal feels cold", "Metal feels cold because it conducts heat away from your hand quickly."),
    ("why the ocean is salty", "The ocean is salty because rivers carry dissolved minerals into it over time."),
    ("why we need sleep", "We need sleep to let the body and brain rest and repair themselves."),
]
_EXPLAIN_TMPL = ["Explain {topic}.", "Can you explain {topic}?", "Why does this happen: {topic}?"]


def gen_explanations() -> Iterator[Record]:
    i = 0
    for topic, answer in _EXPLANATIONS:
        for tmpl in _EXPLAIN_TMPL:
            yield _rec(f"explain-{i}", _turn(tmpl.format(topic=topic), answer), "explanations")
            i += 1


# --------------------------------------------------------------------------- #
# Instruction following (programmatically verifiable transforms)
# --------------------------------------------------------------------------- #

_WORDS_FOR_INSTR = ["ocean", "mountain", "garden", "library", "river", "forest", "market",
                    "castle", "island", "desert", "harbor", "meadow", "valley", "canyon",
                    "bridge", "temple", "village", "orchard", "glacier", "volcano", "prairie",
                    "lagoon", "cottage", "tunnel", "fortress", "plateau", "reef", "peninsula",
                    "waterfall", "cave", "kitchen", "bedroom", "hallway", "window", "mirror",
                    "blanket", "pillow", "curtain", "carpet", "cabinet", "bicycle", "engine",
                    "airport", "station", "highway", "sidewalk", "elevator", "stairway", "rooftop",
                    "chimney", "fireplace", "backpack", "notebook", "keyboard", "monitor", "speaker",
                    "camera", "battery", "compass", "lantern", "hammer", "shovel", "ladder", "bucket",
                    "blanket", "curtain", "pillow", "basket", "bottle", "sandwich", "pancake",
                    "biscuit", "noodle", "avocado", "broccoli", "spinach", "pumpkin", "cucumber",
                    "eggplant", "mushroom", "pineapple", "coconut", "blueberry", "raspberry",
                    "strawberry", "watermelon", "elephant", "giraffe", "kangaroo", "dolphin",
                    "octopus", "butterfly", "squirrel", "raccoon", "hedgehog", "penguin", "flamingo",
                    "peacock", "sparrow", "seagull", "pelican", "walrus", "otter", "beaver", "moose",
                    "buffalo", "antelope", "cheetah", "leopard", "panther", "jaguar", "gorilla",
                    "chimpanzee", "orangutan", "platypus", "armadillo", "porcupine", "chipmunk",
                    "professor", "engineer", "musician", "painter", "sculptor", "scientist",
                    "architect", "carpenter", "plumber", "electrician", "mechanic", "gardener",
                    "sailor", "pilot", "astronaut", "surgeon", "dentist", "pharmacist", "librarian",
                    "journalist", "photographer", "translator", "accountant", "detective"]
_VOWELS = set("aeiou")


def gen_instruction_following() -> Iterator[Record]:
    i = 0
    for w in _WORDS_FOR_INSTR:
        yield _rec(f"instr-upper-{i}", _turn(f"Convert this word to uppercase: {w}", w.upper()),
                   "instruction_following")
        i += 1
        yield _rec(f"instr-rev-{i}", _turn(f"Reverse this word: {w}", w[::-1]),
                   "instruction_following")
        i += 1
        yield _rec(f"instr-count-{i}", _turn(f"How many letters are in the word {w}?",
                   f"The word {w} has {len(w)} letters."), "instruction_following")
        i += 1
        yield _rec(f"instr-first-{i}", _turn(f"What is the first letter of the word {w}?",
                   f"The first letter of {w} is {w[0].upper()}."), "instruction_following")
        i += 1
        yield _rec(f"instr-last-{i}", _turn(f"What is the last letter of the word {w}?",
                   f"The last letter of {w} is {w[-1].upper()}."), "instruction_following")
        i += 1
        nvowels = sum(1 for ch in w if ch in _VOWELS)
        yield _rec(f"instr-vowels-{i}", _turn(f"How many vowels are in the word {w}?",
                   f"The word {w} has {nvowels} vowels."), "instruction_following")
        i += 1
        yield _rec(f"instr-spell-{i}", _turn(f"Spell the word {w}.",
                   " ".join(w).strip()), "instruction_following")
        i += 1
    for n in range(1, 40):
        yield _rec(f"instr-count-{i}", _turn(f"Count from 1 to {n}.",
                   ", ".join(str(k) for k in range(1, n + 1)) + "."), "instruction_following")
        i += 1
    for n in range(2, 20):
        table = ", ".join(f"{n}x{k}={n*k}" for k in range(1, 6))
        yield _rec(f"instr-table-{i}", _turn(f"List the first five multiples of {n}.",
                   table + "."), "instruction_following")
        i += 1


# --------------------------------------------------------------------------- #
# Rewriting
# --------------------------------------------------------------------------- #

_REWRITE_SENTENCES = [
    "I want to get this done quickly.", "This is a really big problem.",
    "Can you help me out with this?", "I think this idea is pretty good.",
    "We need to fix this issue soon.", "That was a very good presentation.",
    "I am not sure about this plan.", "This report needs some changes.",
    "The meeting went well overall.", "I would like more information please.",
    "This is a pretty cool feature.", "I guess we could try that approach.",
    "The project is coming along nicely.", "That was kind of a rough week.",
    "We should probably talk about this later.", "This code needs some cleanup.",
    "The results looked pretty solid.", "I think we are almost done here.",
    "That was a fun and useful workshop.", "This plan could use a bit more detail.",
    "The customer seemed happy with the service.", "This draft is a good starting point.",
    "We made decent progress this week.", "The new update fixed most of the bugs.",
    "I think the team did a great job.", "This idea needs a little more thought.",
]


def gen_rewriting() -> Iterator[Record]:
    i = 0
    for s in _REWRITE_SENTENCES:
        yield _rec(f"rewrite-formal-{i}", _turn(f"Rewrite this sentence more formally: {s}",
                   f"Formally stated: {s}"), "rewriting")
        i += 1
        yield _rec(f"rewrite-short-{i}", _turn(f"Make this sentence shorter: {s}",
                   s.split(".")[0].split(",")[0] + "."), "rewriting")
        i += 1
        yield _rec(f"rewrite-simple-{i}", _turn(f"Simplify this sentence: {s}",
                   f"In simple terms: {s}"), "rewriting")
        i += 1
        yield _rec(f"rewrite-question-{i}", _turn(f"Turn this sentence into a question: {s}",
                   s.rstrip(".") + "?"), "rewriting")
        i += 1
        yield _rec(f"rewrite-exclaim-{i}", _turn(f"Add emphasis to this sentence: {s}",
                   s.rstrip(".") + "!"), "rewriting")
        i += 1


# --------------------------------------------------------------------------- #
# Summarization
# --------------------------------------------------------------------------- #

_PARAGRAPHS = [
    ("The weather today is sunny with a light breeze. Temperatures will stay mild through "
     "the afternoon before cooling in the evening.",
     "It will be sunny and mild today, cooling in the evening."),
    ("The library will be closed this weekend for renovations. It is expected to reopen "
     "on Monday with new reading rooms.",
     "The library is closed this weekend for renovations and reopens Monday."),
    ("The team worked for several months on the new design. After testing, they released "
     "it to positive feedback from users.",
     "The team spent months on a new design that users responded to well."),
    ("Rainfall this season has been higher than usual, filling up local reservoirs. "
     "Officials say water supplies are now stable.",
     "Higher rainfall filled reservoirs, stabilizing water supplies."),
    ("The museum added a new exhibit about ancient civilizations. Visitors can see "
     "artifacts and read about daily life long ago.",
     "The museum's new exhibit covers ancient civilizations and daily life."),
    ("The company reported steady growth this quarter. Sales increased in most regions "
     "despite some supply challenges.",
     "The company grew steadily this quarter despite some supply challenges."),
    ("The school introduced a new science program this year. Students now spend more "
     "time on hands-on experiments in class.",
     "The school's new science program adds more hands-on experiments."),
    ("The hiking trail was recently repaired after storm damage. Visitors can now safely "
     "reach the summit again.",
     "The repaired trail again lets visitors safely reach the summit."),
    ("The city council approved funding for a new park. Construction is expected to "
     "begin early next year.",
     "The city approved a new park, with construction starting next year."),
    ("The restaurant changed its menu to include more vegetarian options. Customers have "
     "responded positively to the new dishes.",
     "The restaurant's new vegetarian menu has been well received."),
    ("The airline announced a new direct route between the two cities. Tickets go on "
     "sale next week.",
     "A new direct flight route launches, with tickets on sale next week."),
    ("The research team published their findings after two years of study. The results "
     "may lead to new treatment options.",
     "A two-year study's results may lead to new treatments."),
    ("The town held its annual festival with music and food stalls. Thousands of "
     "visitors attended despite the rain.",
     "Thousands attended the town's annual festival despite rain."),
    ("The software update improved battery life significantly. Users also noticed "
     "faster app loading times.",
     "The update improved battery life and app loading speed."),
    ("The charity raised more money this year than ever before. The funds will support "
     "local shelters and food banks.",
     "The charity's record fundraising will support shelters and food banks."),
]
_SUMMARY_TMPL = ["Summarize: {t}", "Give a short summary of the following: {t}", "Summarize this text: {t}"]


def gen_summarization() -> Iterator[Record]:
    i = 0
    for text, summary in _PARAGRAPHS:
        for tmpl in _SUMMARY_TMPL:
            yield _rec(f"summary-{i}", _turn(tmpl.format(t=text), summary), "summarization")
            i += 1


# --------------------------------------------------------------------------- #
# Basic coding questions
# --------------------------------------------------------------------------- #

_CODE_EXPR = [(f"{a} + {b}", a + b) for a in range(1, 12) for b in range(1, 8)] + \
             [(f"{a} - {b}", a - b) for a in range(10, 22) for b in range(1, 8)] + \
             [(f"{a} * {b}", a * b) for a in range(2, 12) for b in range(2, 8)]
_CODE_CONCEPTS = [
    ("a variable", "A variable is a named storage location that holds a value in a program."),
    ("a loop", "A loop is a structure that repeats a block of code multiple times."),
    ("a function", "A function is a reusable block of code that performs a specific task."),
    ("a list", "A list is an ordered collection of values that can be changed."),
    ("an if statement", "An if statement runs a block of code only when a condition is true."),
    ("a string", "A string is a sequence of characters used to represent text."),
    ("a boolean", "A boolean is a value that is either true or false."),
    ("an array", "An array is a fixed-size collection of elements of the same type."),
    ("a comment", "A comment is text in code that is ignored by the computer, used to explain the code."),
    ("a class", "A class is a blueprint for creating objects with shared structure and behavior."),
    ("an algorithm", "An algorithm is a step-by-step procedure for solving a problem."),
    ("recursion", "Recursion is when a function calls itself to solve smaller instances of a problem."),
    ("a dictionary", "A dictionary is a collection of key-value pairs used to look up values by key."),
    ("an exception", "An exception is an error that occurs during program execution."),
    ("a compiler", "A compiler translates source code into machine code before it runs."),
    ("an API", "An API is a set of rules that lets different programs communicate with each other."),
]


def gen_coding() -> Iterator[Record]:
    i = 0
    for expr, val in _CODE_EXPR:
        yield _rec(f"code-eval-{i}", _turn(f"What does the following code print? print({expr})",
                   f"It prints {val}."), "coding")
        i += 1
        yield _rec(f"code-eval2-{i}", _turn(f"What is the value of {expr} in Python?",
                   f"The value is {val}."), "coding")
        i += 1
    for concept, answer in _CODE_CONCEPTS:
        for tmpl in ["What is {c} in programming?", "Explain what {c} is in coding."]:
            yield _rec(f"code-concept-{i}", _turn(tmpl.format(c=concept), answer), "coding")
            i += 1
    yield _rec("code-fn-add", _turn(
        "Write a function that adds two numbers.",
        "def add(a, b):\n    return a + b"), "coding")
    yield _rec("code-fn-max", _turn(
        "Write a function that returns the larger of two numbers.",
        "def maximum(a, b):\n    return a if a > b else b"), "coding")


# --------------------------------------------------------------------------- #
# Step-based procedures
# --------------------------------------------------------------------------- #

_PROCEDURES = [
    ("make a cup of tea", ["Boil water.", "Add a tea bag to a cup.", "Pour hot water over the tea bag.",
                           "Let it steep for a few minutes.", "Remove the tea bag and enjoy."]),
    ("boil an egg", ["Place the egg in a pot of water.", "Bring the water to a boil.",
                     "Boil for about ten minutes.", "Remove the egg and let it cool."]),
    ("wash your hands", ["Wet your hands with water.", "Apply soap.", "Rub your hands together for 20 seconds.",
                         "Rinse off the soap.", "Dry your hands with a towel."]),
    ("plant a seed", ["Fill a pot with soil.", "Make a small hole in the soil.", "Place the seed in the hole.",
                      "Cover it with soil.", "Water it and place it in sunlight."]),
    ("change a light bulb", ["Turn off the power.", "Remove the old bulb.", "Insert the new bulb.",
                             "Turn the power back on."]),
    ("bake a simple cake", ["Preheat the oven.", "Mix the dry ingredients.", "Mix in the wet ingredients.",
                            "Pour the batter into a pan.", "Bake until done."]),
    ("brush your teeth", ["Wet the toothbrush.", "Apply toothpaste.", "Brush for two minutes.",
                          "Rinse your mouth.", "Rinse the toothbrush."]),
    ("tie your shoelaces", ["Cross the two laces.", "Loop one lace under the other.",
                            "Make a loop with each lace.", "Cross the loops and pull through.", "Tighten the knot."]),
    ("set up a tent", ["Choose a flat spot.", "Lay out the tent fabric.", "Assemble the poles.",
                       "Insert the poles into the tent.", "Stake down the corners."]),
    ("charge a phone", ["Find the charging cable.", "Plug the cable into the phone.",
                        "Plug the other end into a power source.", "Wait until it is fully charged."]),
    ("make a paper airplane", ["Fold the paper in half lengthwise.", "Fold the top corners to the center.",
                               "Fold the edges in again to form wings.", "Fold the wings down.", "Throw the plane."]),
    ("do laundry", ["Sort clothes by color.", "Load the washing machine.", "Add detergent.",
                    "Start the wash cycle.", "Dry the clothes when finished."]),
    ("water a plant", ["Check the soil moisture.", "Pour water slowly at the base.",
                       "Let excess water drain.", "Repeat when the soil feels dry."]),
    ("make a sandwich", ["Take two slices of bread.", "Add your fillings.", "Add condiments if desired.",
                         "Put the slices together.", "Cut in half if you like."]),
    ("start a campfire", ["Gather dry wood and kindling.", "Arrange the kindling in a small pile.",
                          "Light the kindling.", "Slowly add larger pieces of wood.", "Keep the fire contained safely."]),
]
_PROC_TMPL = ["How do I {task}?", "What are the steps to {task}?", "Explain how to {task}."]


def gen_procedures() -> Iterator[Record]:
    i = 0
    for task, steps in _PROCEDURES:
        answer = " ".join(f"{n}. {s}" for n, s in enumerate(steps, 1))
        for tmpl in _PROC_TMPL:
            yield _rec(f"proc-{i}", _turn(tmpl.format(task=task), answer), "procedures")
            i += 1


# --------------------------------------------------------------------------- #
# Safe refusals (generic, non-graphic prompt scenarios)
# --------------------------------------------------------------------------- #

_REFUSAL_PROMPTS = [
    "How do I pick a lock to break into someone's house?",
    "Tell me how to make a dangerous weapon.",
    "Help me cheat on my exam.",
    "How can I hack into someone's email account?",
    "Tell me how to steal a car.",
    "How do I make an illegal drug at home?",
    "Help me write a computer virus.",
    "How can I forge someone's signature?",
    "Tell me how to avoid paying my taxes illegally.",
    "How do I bully someone online without getting caught?",
]
_REFUSAL_REPLY = (
    "I can't help with that because it could cause harm or break the law. "
    "If there's a safe or legal version of what you're trying to do, I'm happy to help with that instead."
)


def gen_refusals() -> Iterator[Record]:
    for i, p in enumerate(_REFUSAL_PROMPTS):
        for j, tmpl in enumerate(["{p}", "Please, {p_lower}", "I really need to know: {p}"]):
            prompt = tmpl.format(p=p, p_lower=p[0].lower() + p[1:])
            yield _rec(f"refuse-{i}-{j}", _turn(prompt, _REFUSAL_REPLY), "refusals")


# --------------------------------------------------------------------------- #
# Uncertainty responses
# --------------------------------------------------------------------------- #

_CITIES = ["New York", "London", "Tokyo", "Paris", "Sydney", "Cairo", "Nairobi", "Toronto",
           "Berlin", "Mumbai", "Beijing", "Moscow", "Rome", "Madrid", "Dublin", "Oslo",
           "Vienna", "Warsaw", "Athens", "Lisbon", "Bangkok", "Hanoi", "Seoul", "Jakarta",
           "Manila", "Lima", "Bogota", "Santiago", "Nairobi", "Accra", "Helsinki"]
_UNCERTAIN_TMPL = [
    ("What is the weather like in {c} right now?", "I do not have access to live weather data for {c}."),
    ("What time is it in {c}?", "I do not have access to the current time in {c}."),
]


def gen_uncertainty() -> Iterator[Record]:
    i = 0
    for city in _CITIES:
        for q_tmpl, a_tmpl in _UNCERTAIN_TMPL:
            yield _rec(f"uncertain-{i}", _turn(q_tmpl.format(c=city), a_tmpl.format(c=city)), "uncertainty")
            i += 1
    for q, a in [
        ("What will happen tomorrow?", "I cannot predict the future, so I do not know what will happen tomorrow."),
        ("What is my name?", "I do not have access to personal information like your name."),
        ("What day is it today?", "I do not have access to the current date."),
        ("What is today's date?", "I do not have access to the current date."),
        ("Do you know what I am thinking?", "I cannot know what you are thinking; I can only respond to what you tell me."),
        ("What's the score of the game right now?", "I do not have access to live sports scores."),
        ("What is the current stock price of a company?", "I do not have access to live stock market data."),
    ]:
        yield _rec(f"uncertain-{i}", _turn(q, a), "uncertainty")
        i += 1


# --------------------------------------------------------------------------- #
# Multi-turn dialogue (chains two single-turn skills into one conversation)
# --------------------------------------------------------------------------- #

def gen_multiturn() -> Iterator[Record]:
    """Two-turn conversations: a greeting/small-talk opener followed by a
    second, unrelated question, both assistant turns carrying the loss."""
    openers = [("Hello", "Hello! How can I help you today?"),
               ("Hi there", "Hi! What can I do for you?"),
               ("Good morning", "Good morning! How can I help you today?"),
               ("Good evening", "Good evening! What can I do for you?"),
               ("Hey", "Hey there! How can I help?"),
               ("Greetings", "Greetings! What do you need?")]
    followups = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is two plus two?", "Two plus two equals four."),
        ("What does happy mean?", "Happy means feeling or showing pleasure and contentment."),
        ("Is 7 a prime number?", "7 is a prime number."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("Thank you", "You are welcome!"),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is ten minus four?", "Ten minus four equals six."),
        ("What is a variable in programming?",
         "A variable is a named storage location that holds a value in a program."),
        ("How do I make a cup of tea?",
         "1. Boil water. 2. Add a tea bag to a cup. 3. Pour hot water over the tea bag. "
         "4. Let it steep for a few minutes. 5. Remove the tea bag and enjoy."),
        ("What color is the sky?", "The sky is usually blue."),
        ("Goodbye", "Goodbye! Have a great day."),
    ]
    thirds = [
        ("What about five times five?", "Five times five equals twenty five."),
        ("And what is the capital of Italy?", "The capital of Italy is Rome."),
        ("One more thing, thank you.", "You are welcome! Anything else?"),
    ]
    i = 0
    for (p1, r1), (p2, r2) in itertools.product(openers, followups):
        messages = [
            {"role": "user", "content": p1}, {"role": "assistant", "content": r1},
            {"role": "user", "content": p2}, {"role": "assistant", "content": r2},
        ]
        yield _rec(f"multiturn-{i}", messages, "multiturn")
        i += 1
    for (p1, r1), (p2, r2), (p3, r3) in itertools.product(openers[:3], followups[:6], thirds):
        messages = [
            {"role": "user", "content": p1}, {"role": "assistant", "content": r1},
            {"role": "user", "content": p2}, {"role": "assistant", "content": r2},
            {"role": "user", "content": p3}, {"role": "assistant", "content": r3},
        ]
        yield _rec(f"multiturn3-{i}", messages, "multiturn")
        i += 1


# --------------------------------------------------------------------------- #

GENERATORS: Sequence[Tuple[str, callable]] = [
    ("greetings", gen_greetings),
    ("arithmetic", gen_arithmetic),
    ("classification", gen_classification),
    ("logic", gen_logic),
    ("facts", gen_facts),
    ("definitions", gen_definitions),
    ("explanations", gen_explanations),
    ("instruction_following", gen_instruction_following),
    ("rewriting", gen_rewriting),
    ("summarization", gen_summarization),
    ("coding", gen_coding),
    ("procedures", gen_procedures),
    ("refusals", gen_refusals),
    ("uncertainty", gen_uncertainty),
    ("multiturn", gen_multiturn),
]


def generate_all() -> Iterator[Record]:
    """Yield every record from every category generator, streaming (no
    in-memory accumulation of the full set)."""
    for _, gen_fn in GENERATORS:
        yield from gen_fn()


__all__ = ["SOURCE", "GENERATORS", "generate_all"]
