(define (domain virtual-home)
    (:requirements :fluents :adl :typing)
    (:types
        item fixture agent itype - object
    )
    (:predicates
        (itemtype ?i - item ?t - itype)
        (active ?a - agent)
        (next-turn ?a - agent ?b - agent)
        (frozen ?a - agent)
        (nograb ?a - agent)
        (noput ?a - agent)
        (holding ?a - agent ?i - item)
        (next-to ?a - agent ?f - fixture)
        (placed ?i - item ?f - fixture)
        (clear ?i - item)
        (stacked ?i - item ?j - item)
    )
    (:action move
     :parameters (?a - agent ?f1 - fixture ?f2 - fixture)
     :precondition
        (and (active ?a) (not (frozen ?a))
            (next-to ?a ?f1) (not (next-to ?a ?f2)))
     :effect
        (and (next-to ?a ?f2) (not (next-to ?a ?f1))
            (forall (?b - agent) (when (next-turn ?a ?b) (active ?b)))
            (not (active ?a)))
    )
    (:action grab
     :parameters (?a - agent ?i - item ?f - fixture)
     :precondition
        (and (active ?a) (not (frozen ?a)) (not (nograb ?a)) (next-to ?a ?f)
            (not (holding ?a ?i)) (placed ?i ?f) (clear ?i))
     :effect
        (and (holding ?a ?i) (not (placed ?i ?f)) (not (clear ?i))
            (forall (?j - item)
                (when (stacked ?i ?j) (and (clear ?j) (not (stacked ?i ?j)))))
            (forall (?b - agent) (when (next-turn ?a ?b) (active ?b)))
            (not (active ?a)))
    )
    (:action put
     :parameters (?a - agent ?i - item ?f - fixture)
     :precondition
        (and (active ?a) (not (frozen ?a)) (not (noput ?a))
            (holding ?a ?i) (next-to ?a ?f))
     :effect
        (and (not (holding ?a ?i)) (placed ?i ?f) (clear ?i)
            (forall (?b - agent) (when (next-turn ?a ?b) (active ?b)))
            (not (active ?a)))
    )
    (:action stack
     :parameters (?a - agent ?i - item ?j - item ?f - fixture)
     :precondition
        (and (active ?a) (not (frozen ?a)) (not (noput ?a))
            (holding ?a ?i) (next-to ?a ?f) (placed ?j ?f) (clear ?j))
     :effect
        (and (not (holding ?a ?i)) (placed ?i ?f)
            (clear ?i) (stacked ?i ?j) (not (clear ?j))
            (forall (?b - agent) (when (next-turn ?a ?b) (active ?b)))
            (not (active ?a)))
    )
    (:action wait
     :parameters (?a - agent)
     :precondition (active ?a)
     :effect (and (forall (?b - agent) (when (next-turn ?a ?b) (active ?b)))
                (not (active ?a)))
    )
)
