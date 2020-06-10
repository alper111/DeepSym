(define (domain john-domain)
  (:requirements :negative-preconditions :conditional-effects
		 :probabilistic-effects :rewards)
  (:predicates (john-is-dirty) (shower-is-on) (has-soap) (has-soccer-ball))
  (:action clean-john
	   :effect (when (and (shower-is-on) (has-soap) (john-is-dirty))
 		          (and (probabilistic 0.8 (not (john-is-dirty)))
                               (probabilistic 0.4 (not (has-soap))))))

  (:action turn-on-shower
	   :effect (when (not (shower-is-on))
			  (probabilistic 0.8 (shower-is-on))))

  (:action turn-off-shower
	   :effect (when (shower-is-on)
			  (probabilistic 0.8 (not (shower-is-on)))))

  (:action buy-soap
	   :effect (when (not (has-soap))
			  (probabilistic 0.8 (has-soap))))

  (:action buy-soccer-ball
	   :effect (when (not (has-soccer-ball))
			  (probabilistic 0.8 (has-soccer-ball))))

  (:action play-soccer
           :effect (and (when (and (has-soccer-ball) (not (shower-is-on)) (not (john-is-dirty)))
			    (and (probabilistic 0.8 (john-is-dirty))
                                 (probabilistic 0.3 (and (not (has-soccer-ball))
                                                         (decrease (reward) 1))
                                                0.7 (increase (reward) 1))))
                        (when (and (has-soccer-ball) (not (shower-is-on)) (john-is-dirty))
			    (and (probabilistic 0.8 (john-is-dirty))
                                 (probabilistic 0.7 (and (not (has-soccer-ball))
                                                         (decrease (reward) 1))
                                                0.3 (increase (reward) 1)))))))

(define (problem john-problem)
  (:domain john-domain)	
  (:metric maximize (reward)))

