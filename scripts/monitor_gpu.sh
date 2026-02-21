#!/bin/bash
# Quick GPU training monitor
ssh -p 25107 root@212.247.220.172 'wc -l /root/distill_p45b.log; grep "INFO: Ep" /root/distill_p45b.log; tail -2 /root/distill_p45b.log'
