from numpy import *
def contour_value_from_posterior(likelihood_2D, percentile):
    """Get contour value from the 2D marginalized distribution
        The contour will contain percentile/100 fraction of the samples.
        """
    if (percentile < 0.0) | (percentile > 100.0):
        raise Exception, 'percentile must be between 0 and 100.'
    flattened_likelihood = likelihood_2D.flatten()
    sorted_likelihood_args = argsort(flattened_likelihood)[::-1]
    sorted_likelihood_points = flattened_likelihood[sorted_likelihood_args]
    px = cumsum(sorted_likelihood_points)/sum(sorted_likelihood_points)
    credible_region = sorted_likelihood_points[where(px<percentile/100.)]
    return credible_region[-1]
