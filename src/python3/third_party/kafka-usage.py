"""
!/usr/bin/python3.7
-*- coding: UTF-8 -*-
Author: https://github.com/Guanyan1996
         ┌─┐       ┌─┐
      ┌──┘ ┴───────┘ ┴──┐
      │                 │
      │       ───       │
      │  ─┬┘       └┬─  │
      │                 │
      │       ─┴─       │
      │                 │
      └───┐         ┌───┘
          │         │
          │         │
          │         │
          │         └──────────────┐
          │                        │
          │                        ├─┐
          │                        ┌─┘
          │                        │
          └─┐  ┐  ┌───────┬──┐  ┌──┘
            │ ─┤ ─┤       │ ─┤ ─┤
            └──┴──┘       └──┴──┘
                神兽保佑
                代码无BUG!

"""
from typing import List

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from loguru import logger


# TODO: 设置producer max_msg_size
class JinShanYunKafka(object):
    def __init__(self, bootstrap_servers: List):
        self.bootstrap_servers = bootstrap_servers

    def producer(self, massage, topic):
        producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers,
                                 sasl_mechanism="PLAIN",
                                 security_protocol='SASL_PLAINTEXT',
                                 sasl_plain_username="",
                                 sasl_plain_password="",
                                 )
        future = producer.send(topic, massage)
        result = future.get(timeout=60)
        logger.info(result)

    def consumer(self, topics):
        consumer = KafkaConsumer(*topics,
                                 bootstrap_servers=self.bootstrap_servers,
                                 sasl_mechanism="PLAIN",
                                 security_protocol='SASL_PLAINTEXT',
                                 api_version=(0, 10),
                                 sasl_plain_username="",
                                 sasl_plain_password=""
                                 )
        print('consumer start to consuming...')
        for message in consumer:
            logger.info(f"{message.topic}, {message.offset}, {message.key}, {message.partition}")

    def customer_from_timestamp(self, topics):
        consumer = KafkaConsumer(*topics,
                                 bootstrap_servers=self.bootstrap_servers,
                                 sasl_mechanism="PLAIN",
                                 security_protocol="SASL_PLAINTEXT",
                                 api_version=(0, 10),
                                 sasl_plain_username="",
                                 sasl_plain_password=""
                                 )
        consumer.poll()
        tp = TopicPartition(topics, 0)
        rec_in = consumer.offsets_for_times({tp: 1636628400000})
        rec_out = consumer.offsets_for_times({tp: 1636635600000})
        consumer.seek(tp, rec_in[tp].offset)
        consumer.seek(tp, rec_in[tp].offset)  # lets go to the first message in New Year!

        for message in consumer:
            if message.offset >= rec_out[tp].offset:
                break
            msg_key = message.key.decode('utf-8')
