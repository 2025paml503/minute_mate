from jinja2 import Template
from itertools import product

template = Template("""
Explain {{ topic }}.

Requirements:
- Audience: {{ audience }}
- Detail level: {{ detail_level }}
- Word limit: {{ word_limit }}
- Examples: {{ num_examples }}
""")

audiences = [
    "children",
    "college students",
    "software engineers"
]

detail_levels = [
    "brief",
    "moderate",
    "deep"
]

word_limits = [100, 300]
example_counts = [0, 2]

variants = []

for audience, detail, words, examples in product(
    audiences,
    detail_levels,
    word_limits,
    example_counts
):
    variants.append(
        template.render(
            topic="cloud computing",
            audience=audience,
            detail_level=detail,
            word_limit=words,
            num_examples=examples,
        )
    )

print(f"Generated {len(variants)} prompts")