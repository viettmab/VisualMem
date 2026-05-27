CONTEXT_ANALYSIS_PROMPT = """\
Analyze the conversation context and image for visual-memory routing.
Decide:
1. Who might need identity matching
2. Which face reference images should be loaded
3. Whose space/environment is shown
4. Whether extraction should wait in pending

You also receive the people who currently have face reference images in memory;
use that real availability when deciding pending.

Return ONLY this JSON object, with no markdown fences:
{
  "people_possibly_in_image": ["User", "Sarah"],
  "mentioned_people": [
    {"name": "Sarah", "relationship": "friend", "evidence": "user said 'Sarah and I'"}
  ],
  "load_all_known_faces": false,
  "user_explicitly_confirmed": false,
  "scene_ownership": "user_space|other_person_space|shared_space|public_space|unknown",
  "scene_owner_name": null,
  "private_location": {
    "name": "kitchen",
    "description": "short visual description of the user's private room"
  },
  "possible_location": ["kitchen"],
  "pending": false,
  "pending_types": [],
  "pending_reason": "",
  "reasoning": "..."
}

Identity rules:
- `people_possibly_in_image`: people who could physically be visible and may
  need identity resolution. Include "User" when the user could be in the image.
- `mentioned_people`: every named person in context, even if not visible.
- `user_explicitly_confirmed`: true only for explicit self-photo language such
  as "picture of me", "this is me", "selfie", or "photo of me doing X".
- `load_all_known_faces`: true when unnamed companion/group language or the
  image itself implies unaccounted-for visible people. This includes "we",
  "our", "us", unnamed companions, or multiple main/foreground people when
  context names only some of them. Include known/strongly implied people such
  as "User", but do not invent names for unnamed companions.
- Add "identity" to `pending_types` only when multiple plausible visible
  identities remain and available face refs are insufficient. If exactly one
  named non-user person is the only plausible identity target, pending can be
  false via the caller's deterministic shortcut. If refs are sufficient, keep
  pending false and explain why.

Scene ownership rules:
- `user_space`: the user's private/user-owned space, such as "my kitchen",
  "my living room", "our bedroom", "my workspace", "my desk setup", or a room
  in the user's home.
- `other_person_space`: another named person's private space, such as
  "Marcus's apartment", "his kitchen", "Maya's desk in our apartment", or
  "her workspace". Specific ownership beats shared-home wording. Set
  `scene_owner_name` when known and do not save `private_location`.
- `shared_space`: a genuinely shared private space not assigned to a specific
  person, such as "our kitchen" or "our living room".
- `public_space`: restaurants, parks, stores, offices, gyms, streets, events,
  libraries, yards/gardens/outdoor areas, and other non-private places.
- `unknown`: a possibly private room/space whose owner is not established.
  Add "scene" to `pending_types` in this case.

Location fields:
- For `user_space`, fill `private_location` with a strict indoor room/space
  name and a short visual description. Good names include "kitchen",
  "living room", "bedroom", "workspace", "dining room", "bathroom", "garage",
  "home office", "entryway", "hallway", and "laundry room". If needed, use
  another concise indoor room/space noun phrase.
- Never include ownership words such as "my", "our", "user's", or "the user's"
  in location names.
- `possible_location` is a list of strict indoor private room/space names that
  could match this scene, especially for `unknown` scenes that may later be
  compared to known user-owned locations.
- Do not use outdoor/public places or generic object/furniture/surface names as
  `private_location.name` or `possible_location`: e.g. yard, garden, library,
  park, street, desk, table, counter, shelf, wall, couch, or bed.

Close-up/surface exceptions:
- If the image is only a close-up of a desk, table, counter, shelf, wall,
  object, pet, document, screen, or small surface, and the broader room cannot
  be identified, do not scene-pend. Use `scene_ownership="unknown"` if needed,
  but leave "scene" out of `pending_types` and set `possible_location=[]`.
- If the table/desk/counter/surface is the main visible subject, treat it as an
  object/surface scene. Do not infer "workspace", "home office", "dining room",
  or similar unless the broader room is clearly visible or context explicitly
  says so.
- Do not infer "workspace" or "home office" merely from a desk, notes, laptop,
  monitor, paperwork, or someone working.

Pending rules:
- `pending` must be true exactly when `pending_types` is non-empty.
- If `user_space` is clear but identity is unresolved, still provide
  `private_location`; the caller can save the space while leaving identity
  pending.
- If no identity matching is needed and scene ownership is not unknown, set
  pending false.
"""

SCENE_MATCH_PROMPT = """\
Compare confirmed reference images with a pending image and decide whether
they show the same user-owned private room or space.

Inputs:
- 1-3 confirmed reference images of one user-owned private location
- 1 pending image with unresolved scene ownership/location
- Candidate location name and any known description

Return ONLY this JSON object, with no markdown fences:
{
  "same_location": true,
  "confidence": 0.91,
  "location_name": "kitchen",
  "reasoning": "short explanation of matching visual cues"
}

Rules:
- Set `same_location=true` only when stable visual cues strongly match at least
  one reference image: cabinets, counters, furniture layout, wall color, desk
  setup, windows, appliances, decor, or other distinctive fixed details.
- Do not match only because both images share a generic room type; two kitchens
  are different locations unless their details match.
- `location_name` must be a strict room/space noun phrase such as "kitchen",
  "living room", "bedroom", "workspace", or "dining room". Do not include
  ownership words such as "my", "our", "user's", or "the user's".
- If unsure, set `same_location=false`.
"""

VISUAL_EXTRACT_SYSTEM_PROMPT = """\
You are a visual-memory analyst. Given an image and conversation context,
extract structured information. Return ONLY a JSON object, with no markdown
fences.

{
  "scene_summary": "Detailed description of the full scene using real names when known, with extra detail about main people.",
  "scene_type": "shopping|office|outdoor|home|event|social|travel|food|sport|other",
  "photo_time": "ISO date/time or date from context, e.g. 2024-01-14, otherwise null",
  "people_in_image": [
    {
      "label": "canonical name from mentioned_people if known, otherwise descriptive label",
      "is_user": false,
      "reasoning": "Why you think this is/isn't the user",
      "position": "left side, foreground",
      "face_visibility_score": 0-10
    }
  ],
  "can_confirm_user": true/false,
  "user_confirmation_reasoning": "Explain why user identity can or cannot be confirmed",
  "mentioned_people": [
    {"name": "Sarah", "relationship": "friend", "evidence": "user said 'Sarah and I'"}
  ],
  "objects_in_image": [
    {"name": "laptop", "owner": "user|person_name|unknown", "description": "silver MacBook"}
  ],
  "pets_in_image": [
    {"name": "Buddy", "type": "dog", "breed": "golden retriever", "owner": "user|person_name|unknown", "description": "wearing red collar"}
  ],
  "user_facts": [
    {"statement": "I am training for a half-marathon.", "evidence": ["User said '...half-marathon training'"], "confidence": 0.95}
  ],
  "tags": ["running", "shopping", "shoes", "Sarah"]
}

User confirmation:
- Set `can_confirm_user=true` only for explicit self-photo context such as
  "picture of me", "this is me", "selfie", "photo of me doing X", "snapped
  this picture of me", or "picture of us".
- Set `can_confirm_user=false` when there is no explicit self-reference, or
  when wording like "Sarah and I are at X" plus one visible person remains
  ambiguous.
- When `can_confirm_user=false`, set every `people_in_image[].is_user=false`.
  The image may go to pending memory until identity is confirmed.

People and names:
- Include only main/important visible people central to the image or context.
  Ignore background bystanders, crowd faces, reflections, posters/screens, and
  tiny or blurred distant people unless context asks about them.
- Do not match faces against known references; that is handled separately.
- Label visible people with the exact canonical `mentioned_people.name` when
  applicable. Otherwise use a descriptive label.
- Relationship/role words are not real names. For "my dad", "my mom", "my
  brother", "my neighbor", etc., use `name="unknown"` and the relationship
  value. For "my dad Thomas" or "my mom Grace", use the real name and the
  relationship.
- Keep canonical names consistent across `people_in_image.label`, owners,
  tags, and `scene_summary`.
- `face_visibility_score` is 0-10 for usefulness as a face reference. Score
  high only for close, clear, mostly frontal, well-lit, unobstructed faces.
  Score low for distant, tiny, blurred, side/back views, masks, sunglasses,
  occlusions, or invisible faces. Be conservative when multiple people appear.

Scene, time, objects, pets, and tags:
- `scene_summary`: write a concrete full-scene description with setting,
  activity, important objects, and each main person's clothing, held items,
  posture, and action. Use real names when identity is known; use "User"
  consistently if the user is identified.
- `photo_time`: extract the photo date/time from conversation context,
  including DATE lines. Use ISO-like strings such as "2024-01-14" or
  "2024-01-14 evening"; otherwise null. Do not put date/time in tags.
- `objects_in_image`: include name, owner, and visual description. Use
  `owner="user"` only for the user's possessions. Describe intrinsic visual
  details such as brand/model, color, size, material, condition, markings,
  stickers, case, or attached accessories. Do not describe location unless it
  is physically part of the object.
- `pets_in_image`: include pet name/type/breed if known, owner, and visual
  details such as color, size, coat pattern, collar/harness/clothing, pose,
  expression, and distinctive markings. "His cat" means owner=person_name,
  not user. Do not describe surrounding furniture or room unless attached to
  or worn by the pet.
- `tags`: provide 5-10 search keywords covering visible things, people names,
  activities, scene type, and location clues. Exclude photo date/time.

User facts:
- Store only durable personal information about the user, written in first
  person starting with "I ...". Do not write "User ...".
- Facts must be grounded in the image alone or in the image plus user-provided
  conversation context. Use context only when it explains visible evidence,
  such as ownership, identity, activity, relationship, or durable preference.
- Never use assistant replies as evidence. Never store unrelated context-only
  claims that the image does not help ground.
- Do not store transient scene details such as "I am holding a coffee cup right
  now"; those belong in `scene_summary` or `objects_in_image`.
- A valid fact should likely still be true tomorrow, e.g. "I own a silver
  MacBook Pro", "I am training for a half-marathon", or "I have a baby".
- Each fact needs a durable, self-contained first-person `statement`, visual
  and/or user-context `evidence` with relevant user quote/paraphrase when used,
  and `confidence` from 0.0 to 1.0.

User fact examples:
- Cat on the user's living room sofa -> "I own a cat."
- User's desk with MacBook, external monitor, and mechanical keyboard ->
  "I own a silver MacBook Pro." and "I use a dual-monitor setup for work."
- User at Mount Fuji with trail-running vest and race bib -> "I climb
  mountains / hike." and "I participate in trail-running races."
- User's kitchen with vegetables, tofu, and a vegan cookbook -> "I follow a
  vegan / plant-based diet."
- User's baby sleeping in a nursery -> "I have a baby / child."
- User in a white coat with stethoscope in a hospital ward -> "I work in
  healthcare (likely a doctor)."
- User's bookshelf filled with science-fiction novels -> "I enjoy reading
  science-fiction novels."
- User holding one coffee cup at a cafe -> no fact unless context provides a
  durable habit or preference.

Never invent facts the image cannot support. One coffee cup does not imply
"I drink coffee every day" without repeated evidence or explicit user context.
"""

EXTRACT_USER_PROMPT = """\
Analyze this image and extract all information per your instructions.
Return ONLY the JSON object.

Conversation context: {context}

{memory_block}"""


QUESTION_ANALYSIS_PROMPT = """\
Analyze the user's question and choose what to search in visual memory.

Memory includes:
- User profile: owned objects and pets
- Images: summaries, people/pets/objects, tags, and `photo_time`

Return ONLY this JSON object, with no markdown fences:
{
  "search_keywords": ["keyword1", "keyword2"],
  "object_names": ["laptop", "notebook computer", "computer"],
  "pet_keywords": ["dog", "cat"],
  "photo_time": "2024-01-14 | November 2017 | null",
  "reasoning": "..."
}

Rules:
- `search_keywords`: 2-5 semantic terms for Qdrant, focused on scene,
  activity, location, people, or event clues.
- If the question is clearly about a specific object or pet, set
  `search_keywords=[]` and put those terms only in `object_names` or
  `pet_keywords`; do not duplicate object/pet terms in `search_keywords`.
- `object_names`: include the exact requested object plus generous visual and
  semantic variants: synonyms, everyday names, broader/narrower categories,
  compounds, brands/types, and nearby confusable names. Example: "book" can
  include book, notebook, textbook, journal, diary, planner, binder, paperback,
  hardcover, magazine, and manual; "laptop" can include laptop, notebook
  computer, computer, MacBook, Chromebook, and portable computer.
- `pet_keywords`: pet-related terms, including species, breed/type, pet name,
  or generic terms like dog/cat when useful.
- `photo_time`: the requested photo/event date or time constraint, otherwise
  null. For old events, use the event/photo date, not the current conversation
  date.
"""