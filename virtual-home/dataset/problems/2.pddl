(define (problem virtual-home-2)
  (:domain virtual-home)
  (:objects 
    cutleryfork cutleryknife plate bowl - itype
    cutleryfork1 cutleryfork2 cutleryfork3 cutleryfork4
    cutleryknife1 cutleryknife2 cutleryknife3 cutleryknife4
    plate1 plate2 plate3 plate4
    bowl1 bowl2 bowl3 bowl4 - item
    table1 cabinet1 cabinet2 cabinet3 cabinet4 cabinet5 - fixture
    actor helper - agent
  )
  (:init (active actor)
         (next-turn actor helper)
         (next-turn helper actor)
         (next-to helper table1)
         (next-to actor table1)
         (itemtype cutleryknife1 cutleryknife)
         (itemtype cutleryknife2 cutleryknife)
         (itemtype cutleryknife3 cutleryknife)
         (itemtype cutleryknife4 cutleryknife)
         (itemtype cutleryfork1 cutleryfork)
         (itemtype cutleryfork2 cutleryfork)
         (itemtype cutleryfork3 cutleryfork)
         (itemtype cutleryfork4 cutleryfork)
         (itemtype plate1 plate)
         (itemtype plate2 plate)
         (itemtype plate3 plate)
         (itemtype plate4 plate)
         (itemtype bowl1 bowl)
         (itemtype bowl2 bowl)
         (itemtype bowl3 bowl)
         (itemtype bowl4 bowl)
         (placed bowl1 cabinet1)
         (placed bowl2 cabinet1)
         (placed bowl3 cabinet1)
         (placed bowl4 cabinet1)
         (clear bowl1)
         (stacked bowl1 bowl2)
         (stacked bowl2 bowl3)
         (stacked bowl3 bowl4)
         (placed plate1 cabinet2)
         (placed plate2 cabinet2)
         (placed plate3 cabinet2)
         (placed plate4 cabinet2)
         (clear plate1)
         (stacked plate1 plate2)
         (stacked plate2 plate3)
         (stacked plate3 plate4)         
         (placed cutleryfork1 cabinet4)
         (placed cutleryfork2 cabinet4)
         (placed cutleryfork3 cabinet4)
         (placed cutleryfork4 cabinet4)
         (clear cutleryfork1)
         (stacked cutleryfork1 cutleryfork2)
         (stacked cutleryfork2 cutleryfork3)
         (stacked cutleryfork3 cutleryfork4)
         (placed cutleryknife1 cabinet4)
         (placed cutleryknife2 cabinet4)
         (placed cutleryknife3 cabinet4)
         (placed cutleryknife4 cabinet4)
         (clear cutleryknife1)
         (stacked cutleryknife1 cutleryknife2)
         (stacked cutleryknife2 cutleryknife3)
         (stacked cutleryknife3 cutleryknife4)
  )
  (:goal (placed table1 plate))
)