;; Bomb and toilet domain:
;;
;;   Kushmerick, Nicholas, Steve Hanks, and Daniel S. Weld.  1995.  An
;;   algorithm for probabilistic planning.  Artificial Intelligence,
;;   76(1-2):239-286.

(define (domain bomb-and-toilet)
  (:requirements :conditional-effects :probabilistic-effects)
  (:predicates (bomb-in-package ?pkg) (toilet-clogged) (bomb-defused))
  (:action dunk-package
	   :parameters (?pkg)
	   :effect (and (when (bomb-in-package ?pkg)
			  (bomb-defused))
			(probabilistic 0.05 (toilet-clogged)))))
