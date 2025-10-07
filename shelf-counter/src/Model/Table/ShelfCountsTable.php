<?php
declare(strict_types=1);

namespace App\Model\Table;

use Cake\ORM\Query\SelectQuery;
use Cake\ORM\RulesChecker;
use Cake\ORM\Table;
use Cake\Validation\Validator;

/**
 * ShelfCounts Model
 *
 * @method \App\Model\Entity\ShelfCount newEmptyEntity()
 * @method \App\Model\Entity\ShelfCount newEntity(array $data, array $options = [])
 * @method array<\App\Model\Entity\ShelfCount> newEntities(array $data, array $options = [])
 * @method \App\Model\Entity\ShelfCount get(mixed $primaryKey, array|string $finder = 'all', \Psr\SimpleCache\CacheInterface|string|null $cache = null, \Closure|string|null $cacheKey = null, mixed ...$args)
 * @method \App\Model\Entity\ShelfCount findOrCreate($search, ?callable $callback = null, array $options = [])
 * @method \App\Model\Entity\ShelfCount patchEntity(\Cake\Datasource\EntityInterface $entity, array $data, array $options = [])
 * @method array<\App\Model\Entity\ShelfCount> patchEntities(iterable $entities, array $data, array $options = [])
 * @method \App\Model\Entity\ShelfCount|false save(\Cake\Datasource\EntityInterface $entity, array $options = [])
 * @method \App\Model\Entity\ShelfCount saveOrFail(\Cake\Datasource\EntityInterface $entity, array $options = [])
 * @method iterable<\App\Model\Entity\ShelfCount>|\Cake\Datasource\ResultSetInterface<\App\Model\Entity\ShelfCount>|false saveMany(iterable $entities, array $options = [])
 * @method iterable<\App\Model\Entity\ShelfCount>|\Cake\Datasource\ResultSetInterface<\App\Model\Entity\ShelfCount> saveManyOrFail(iterable $entities, array $options = [])
 * @method iterable<\App\Model\Entity\ShelfCount>|\Cake\Datasource\ResultSetInterface<\App\Model\Entity\ShelfCount>|false deleteMany(iterable $entities, array $options = [])
 * @method iterable<\App\Model\Entity\ShelfCount>|\Cake\Datasource\ResultSetInterface<\App\Model\Entity\ShelfCount> deleteManyOrFail(iterable $entities, array $options = [])
 *
 * @mixin \Cake\ORM\Behavior\TimestampBehavior
 */
class ShelfCountsTable extends Table
{
    /**
     * Initialize method
     *
     * @param array<string, mixed> $config The configuration for the Table.
     * @return void
     */
    public function initialize(array $config): void
    {
        parent::initialize($config);

        $this->setTable('shelf_counts');
        $this->setDisplayField('key');
        $this->setPrimaryKey('id');

        $this->addBehavior('Timestamp');
    }

    /**
     * Default validation rules.
     *
     * @param \Cake\Validation\Validator $validator Validator instance.
     * @return \Cake\Validation\Validator
     */
    public function validationDefault(Validator $validator): Validator
    {
        $validator
            ->scalar('key')
            ->maxLength('key', 64)
            ->requirePresence('key', 'create')
            ->notEmptyString('key')
            ->add('key', 'unique', ['rule' => 'validateUnique', 'provider' => 'table']);

        $validator
            ->integer('count')
            ->notEmptyString('count');

        return $validator;
    }

    /**
     * Returns a rules checker object that will be used for validating
     * application integrity.
     *
     * @param \Cake\ORM\RulesChecker $rules The rules object to be modified.
     * @return \Cake\ORM\RulesChecker
     */
    public function buildRules(RulesChecker $rules): RulesChecker
    {
        $rules->add($rules->isUnique(['key']), ['errorField' => 'key']);

        return $rules;
    }
}
