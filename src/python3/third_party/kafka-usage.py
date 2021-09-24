import ssl
from typing import List

from kafka import KafkaConsumer, KafkaProducer
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
