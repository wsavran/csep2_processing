
    def get_inter_event_times(self, scale=1000):
        """ Returns interevent times of events in catalog

        Args:
            scale (int): scales epoch times to another unit. default is seconds

        Returns:
            inter_times (ndarray): inter event times

        """
        times = self.get_epoch_times()
        inter_times = numpy.diff(times) / scale
        return inter_times

    def get_inter_event_distances(self, ellps='WGS84'):
        """ Compute distance between successive events in catalog

        Args:
            ellps (str): ellipsoid to compute distances. see pyproj.Geod for more info

        Returns:
            inter_dist (ndarray): ndarray of inter event distances in kilometers
        """
        geod = pyproj.Geod(ellps=ellps)
        lats = self.get_latitudes()
        lons = self.get_longitudes()
        # in-case pyproj doesn't behave nicely all the time
        if self.get_number_of_events() == 0:
            return numpy.array([])
        _, _, dists = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        return dists

    def get_bvalue(self, reterr=True):
        """
        Estimates the b-value of a catalog from Marzocchi and Sandri (2003)

        Args:
            reterr (bool): returns errors

        Returns:
            bval (float): b-value
            err (float): std. err
        """
        if self.get_number_of_events() == 0:
            return None
        # this might fail if magnitudes are not aligned
        mws = discretize(self.get_magnitudes(), CSEP_MW_BINS)
        dmw = CSEP_MW_BINS[1] - CSEP_MW_BINS[0]
        # compute the p term from eq 3.10 in marzocchi and sandri [2003]
        def p():
            top = dmw
            bottom = numpy.mean(mws) - numpy.min(mws)
            # this might happen if all mags are the same, or 1 event in catalog
            if bottom == 0:
                return None
            return 1 + top / bottom

        bottom = numpy.log(10)*dmw
        p = p()
        if p is None:
            return None
        bval = 1.0 / bottom * numpy.log(p)
        if reterr:
            err = (1 - p) / (numpy.log(10) * dmw * numpy.sqrt(self.event_count * p))
            return (bval, err)
        else:
            return bval
