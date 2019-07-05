#include "hiredis/hiredis.h"
#include <stdio.h>
#include <stdlib.h>

redisContext *redis_connect;

/*
 * store key and value to redis
 * Input: key and value
 * Output: if success, return 1; otherwise return 0
 */
int store_to_redis(char key[], char value[]) {
	redisReply *redis_reply = (redisReply *)redisCommand(redis_connect, "SET %s %s", key, value);
	if (redis_reply != NULL) {
		if (redis_reply->type != REDIS_REPLY_ERROR) {
			freeReplyObject(redis_reply);
			return 1;
		}
		freeReplyObject(redis_reply);
	}
	return 0;
}

/* 
 * build a redis connection and store key-value pair to it
 * Input: key-value pair
 * Output: whether store is successful
 */
int main() {

	redis_connect = redisConnect("localhost", 6379);
	if (redis_connect != NULL && redis_connect->err) {
		printf("connect err\n");
		exit(1);
	}

	printf("Please input key and value:\n");
	char key[100];
	char value[100];
	scanf("%s %s\n", key, value); // bad: need to check the return value
	if (!store_to_redis(key, value)) {
		printf("Store Failed!\n");
		exit(1);
	}
	printf("Store Succeed!\n");
	return 0;
}
